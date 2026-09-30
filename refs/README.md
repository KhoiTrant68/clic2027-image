# Repo tham khảo

Các repo này không được commit vào repo chính (xem `.gitignore`). Clone lại vào `refs/repos/<tên>` bằng URL và commit dưới đây.

- `AEIC`: https://github.com/LuizScarlet/AEIC.git @ 7222c2c
- `DiffO`: https://github.com/Freemasti/DiffO.git @ dd56f49
- `MeanFlow`: https://github.com/haidog-yaqub/MeanFlow.git @ 97cc518
- `OSCAR`: https://github.com/jp-guo/OSCAR.git @ cb23f4e
- `RectifiedFlow`: https://github.com/gnobitab/RectifiedFlow.git @ 5a1fd4d
- `StableCodec`: https://github.com/LuizScarlet/StableCodec.git @ b19401c
- `alphaflow`: https://github.com/snap-research/alphaflow.git @ b0fef77
- `conditional-flow-matching`: https://github.com/atong01/conditional-flow-matching.git @ 7c65385
- `onedc`: https://github.com/onedc-codec/onedc.git @ df37891
- `clic-devkit`: https://github.com/clic-challenge/devkit @ 7be3ffb (Docker, requirements và decoder VVC 23.8 của server CLIC)

Ghi chú khi đọc code của đối thủ nằm ở `code_notes.md`.
- `diffusers_src`: 7 file nguồn của diffusers **v0.40.0** (autoencoder_dc, sana_transformer, attention_processor, normalization, embeddings, activations, attention), tải từ https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/src/diffusers/models/. Đây là bản gốc để port `src/ratflow/nn`. Config SANA 0.6B/1.6B cũng nằm ở đây
