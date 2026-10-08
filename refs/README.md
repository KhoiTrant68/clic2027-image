# Repo tham khảo

Các repo này không được commit vào repo chính (xem `.gitignore`). Clone lại vào `refs/repos/<tên>` bằng URL và commit dưới đây.

- `AEIC`: https://github.com/LuizScarlet/AEIC.git @ 7222c2c
- `DiffO`: https://github.com/Freemasti/DiffO.git @ dd56f49
- `OSCAR`: https://github.com/jp-guo/OSCAR.git @ cb23f4e
- `StableCodec`: https://github.com/LuizScarlet/StableCodec.git @ b19401c
- `onedc`: https://github.com/onedc-codec/onedc.git @ df37891
- `NeuralCompression`: https://github.com/facebookresearch/NeuralCompression.git @ 3f12280 (MS-ILLM, ứng viên C2 của bake-off)
- `clic-devkit`: https://github.com/clic-challenge/devkit @ 7be3ffb (Docker, requirements và decoder VVC 23.8 của server CLIC)

Ghi chú khi đọc code của đối thủ nằm ở `code_notes.md`.
- `diffusers_src`: 7 file nguồn của diffusers **v0.40.0** (autoencoder_dc, sana_transformer, attention_processor, normalization, embeddings, activations, attention), tải từ https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/src/diffusers/models/. Đây là bản gốc để port `src/ratflow/nn`. Config SANA 0.6B/1.6B cũng nằm ở đây
