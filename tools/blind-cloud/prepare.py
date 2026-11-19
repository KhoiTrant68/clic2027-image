"""Package a two-codec study for the Worker, keeping labels and codes out of assets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
import shutil


def replace_once(text, original, replacement):
    if text.count(original) != 1:
        raise ValueError(f'Unexpected layout in the shared blind-scoring page: {original[:70]}')
    return text.replace(original, replacement)


def render_page():
    page = (Path(__file__).resolve().parents[2] / 'src/clic27/blind_score.html').read_text(encoding='utf-8')
    welcome = page[page.index('<section id="welcome"'):page.index('<section id="study"')]
    page = replace_once(page, welcome, '''
<section id="welcome" class="welcome">
  <h2>Chọn đợt chấm ảnh</h2>
  <p>Xem ảnh gốc, rồi chọn bản A hoặc B có chất lượng tốt hơn. Nếu khó phân biệt, chọn “Hai bản tương đương”. Tên model được ẩn trong lúc chấm.</p>
  <p class="muted">Không cần tài khoản hay mã truy cập. Quay lại bằng cùng trình duyệt để tiếp tục.</p>
  <p id="round-status" role="status">Đang tải danh sách đợt chấm…</p>
  <div id="round-list"></div>
  <p><a href="/results">Xem bảng kết quả công khai</a></p>
</section>
''')
    page = replace_once(page, '</style>', '''
#round-list{display:grid;gap:14px}.round-card{padding:18px;border:1px solid #41526c;border-radius:12px;background:#111a27}
.round-card h3,.round-card p{margin:0 0 8px}.round-card ul{margin:6px 0 16px;padding-left:20px}
.round-card li{margin:2px 0}.round-card .actions{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.welcome a,#finished a{color:#9cc6ff}.round-card:target{border-color:#9cc6ff}
</style>''')
    page = replace_once(page, 'Kết quả và tên codec được giữ ở phía người tổ chức.',
        'Bạn có thể xem các số đo tại <a href="/results" style="color:#9cc6ff">bảng kết quả</a>.')
    page = replace_once(page, "let token='',key='',trial=null", 'let trial=null')
    page = replace_once(page, "const headers={'Authorization':'Bearer '+token};", 'const headers={};')
    page = replace_once(page, "async function api(path,body){", """let activeRound;
async function api(path,body){
 if(activeRound && path !== '/api/rounds') path += '?round=' + encodeURIComponent(activeRound);""")
    start = page.index("$('#start').onclick")
    end = page.index("</script>", start)
    page = page[:start] + """
function element(tag,text){
 const node=document.createElement(tag);
 node.textContent=text;
 return node;
}
async function showRounds(){
 const {rounds,closed}=await api('/api/rounds');
 const status=$('#round-status');
 status.textContent=closed?'Đợt chấm hiện đã đóng.':rounds.length?'':'Hiện chưa có đợt chấm.';
 const selected=new URLSearchParams(location.search).get('round');
 if(selected && !rounds.some(round=>round.id===selected)) status.textContent='Không tìm thấy đợt trong liên kết. Bạn có thể chọn đợt khác.';
 rounds.sort((a,b)=>Number(b.id===selected)-Number(a.id===selected));
 for(const {id,rates,pairs,progress,questions} of rounds){
  const card=element('section','');
  card.className='round-card';
  card.append(element('h3',id==='clic30-v2'?'CLIC30 · Bốn phiên bản MLIC++':id));
  card.append(element('p',rates.map(rate=>rate.toLocaleString('vi-VN')).join(' / ')+' bpp'));
  card.append(element('p',progress?`${progress.completed}/${progress.total} câu đã lưu`:`${questions} câu mỗi lượt`));
  const list=element('ul','');
  for(const pair of pairs) list.append(element('li',pair.join(' ↔ ')));
  card.append(list);
  const actions=element('div','');
  actions.className='actions';
  const done=progress && progress.completed===progress.total;
  const button=element('button',done?'Đã chấm xong':progress?'Tiếp tục chấm':'Bắt đầu chấm');
  button.className='primary';
  button.disabled=closed||Boolean(done);
  button.onclick=async()=>{
   const buttons=[...$('#round-list').querySelectorAll('button:not(:disabled)')];
   buttons.forEach(item=>item.disabled=true);
   activeRound=id;
   try{
    await api('/api/start',{});
    await next();
   }catch(e){
    $('#error').textContent=e.message;
    if($('#welcome').hidden) return;
    activeRound=undefined;
    buttons.forEach(item=>item.disabled=false);
   }
  };
  const link=element('a','Liên kết đợt này');
  link.href='/?round='+encodeURIComponent(id);
  actions.append(button,link);
  card.append(actions);
  $('#round-list').append(card);
 }
}
$('#question').textContent='So với ảnh gốc, bản nào có chất lượng thị giác tốt hơn?';
$('#finished').append(Object.assign(element('a','Quay lại danh sách đợt chấm'),{href:'/'}));
showRounds().catch(e=>{$('#round-status').textContent=e.message;});
""" + page[end:]
    return page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--people', type=int, default=20)
    parser.add_argument('--trials', type=int, default=15)
    parser.add_argument('--page-only', action='store_true', help='Refresh the HTML of an existing deployment.')
    args = parser.parse_args()
    if args.page_only:
        page_path = args.out / 'public/index.html'
        if not page_path.is_file():
            parser.error('--page-only requires an existing deployment with public/index.html')
        page_path.write_text(render_page(), encoding='utf-8')
        return
    if args.study is None:
        parser.error('--study is required when creating a deployment')

    manifest = json.loads((args.study / 'private_manifest.json').read_text(encoding='utf-8'))
    codecs = manifest['codecs']
    images = manifest['images'][:]
    if len(codecs) != 2 or not 1 <= args.trials <= len(images) or not 1 <= args.people <= 100:
        raise ValueError('Need exactly two codecs, unique images per invitation, and 1–100 invitations.')
    asset_ids = {asset for image in images for asset in [image['reference'], *image['codecs'].values()]}
    for asset in asset_ids:
        if not re.fullmatch(r'[a-f0-9]{48}', asset):
            raise ValueError('Invalid asset identifier.')
        if (args.study / 'assets' / f'{asset}.png').stat().st_size > 25 * 1024**2:
            raise ValueError('Asset exceeds the Workers Free per-file limit.')
    page = render_page()

    args.out.mkdir(parents=True, exist_ok=False)
    asset_dir = args.out / 'public/asset'
    asset_dir.mkdir(parents=True)
    for asset in asset_ids:
        name = f'{asset}.png'
        shutil.copyfile(args.study / 'assets' / name, asset_dir / name)
    (args.out / 'public/index.html').write_text(page, encoding='utf-8')
    rng = random.SystemRandom()
    rng.shuffle(images)
    flips = {image['source']: rng.randrange(2) for image in images}
    with (args.out / 'seed.sql').open('w', encoding='utf-8') as sql, \
            (args.out / 'invitations.csv').open('w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['participant', 'invitation'])

        def insert(table, values):
            fields = ', '.join("'" + str(value).replace("'", "''") + "'" for value in values)
            sql.write(f'INSERT INTO {table} VALUES ({fields});\n')

        for i in range(args.people):
            participant = f'P{i+1:02}'
            code = secrets.token_hex(16)
            writer.writerow([participant, code])
            insert('invitations', [hashlib.sha256(code.encode()).hexdigest(), participant])
            assigned = []
            for index in range(i * args.trials, (i + 1) * args.trials):
                image = images[index % len(images)]
                reverse = (index // len(images) + flips[image['source']]) % 2
                a, b = codecs[::-1] if reverse else codecs
                assigned.append([image['source'], image['reference'], image['codecs'][a], image['codecs'][b], a, b])
            rng.shuffle(assigned)
            for position, trial in enumerate(assigned):
                insert('trials', [secrets.token_hex(16), participant, position, *trial])
    print(f'{len(asset_ids)} images, {args.people} invitations, {args.trials} pairs per invitation')


if __name__ == '__main__':
    main()
