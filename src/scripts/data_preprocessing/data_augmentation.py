import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A


def save_transforms_side_by_side(
    image_path: str,
    output_path: str,
    size=(224, 224),
    grid_rows=2,
    grid_cols=3,
):
    """
    Gera uma imagem única em grid 2x3 com 6 tiles:
    [ original | hflip | vflip ]
    [ blur     | dropout | hsv_shift ]

    - image_path: caminho da imagem original (PNG/JPG)
    - output_path: caminho do arquivo PNG resultante
    - size: tamanho (H, W) para redimensionar todas as imagens
    - grid_rows, grid_cols: dimensões do grid (padrão 2x3)
    """

    # -------------------------
    # 1. Carrega imagem original
    # -------------------------
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    target_h, target_w = size
    img_resized = cv2.resize(img_np, (target_w, target_h))  # cv2 usa (W,H)

    # -------------------------
    # 2. Define exatamente 6 transformações (inclui original)
    # -------------------------
    transforms_dict = {
        "original": None,  # sem transformação
        "hflip": A.HorizontalFlip(p=1.0),
        "vflip": A.VerticalFlip(p=1.0),
        "blur": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        "dropout": A.CoarseDropout(
            max_holes=5,
            max_height=8,
            max_width=8,
            p=1.0
        ),
        "hsv_shift": A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=1.0
        ),
        # NOTE: mantivemos apenas 6 entradas no dicionário
    }

    tiles = []

    # -------------------------
    # 3. Aplica cada transformação e monta os “tiles”
    # -------------------------
    for name, aug in transforms_dict.items():
        if aug is None:
            # Já redimensionado acima
            tile_rgb = img_resized.copy()
        else:
            # Compose só com Resize + transformação específica
            comp = A.Compose([
                A.Resize(target_h, target_w),
                aug,
            ])
            tile_rgb = comp(image=img_np)["image"]

        # Garante que é uint8
        tile_rgb = np.clip(tile_rgb, 0, 255).astype(np.uint8)

        # Converte para BGR para desenhar texto e salvar com OpenCV
        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

        # -------------------------
        # 4. Escreve o nome da transformação
        # -------------------------
        text = name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Fundo levemente escuro atrás do texto (para melhorar contraste)
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        # Caixa no topo
        cv2.rectangle(
            tile_bgr,
            (0, 0),
            (text_w + 6, text_h + 6),
            (0, 0, 0),
            thickness=-1
        )
        # Texto em branco
        cv2.putText(
            tile_bgr,
            text,
            (3, text_h + 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA
        )

        tiles.append(tile_bgr)

    # -------------------------
    # 5. Garante ter exatamente grid_rows * grid_cols tiles
    # -------------------------
    expected = grid_rows * grid_cols
    if len(tiles) > expected:
        tiles = tiles[:expected]
    elif len(tiles) < expected:
        # preenche com blocos pretos se necessário
        h, w = tiles[0].shape[:2]
        black = np.zeros((h, w, 3), dtype=np.uint8)
        while len(tiles) < expected:
            tiles.append(black)

    # -------------------------
    # 6. Monta o grid (2x3)
    # -------------------------
    rows = []
    for r in range(grid_rows):
        start = r * grid_cols
        end = start + grid_cols
        row_tiles = tiles[start:end]
        row = np.concatenate(row_tiles, axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)

    # Cria diretório de saída, se necessário
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva imagem final
    cv2.imwrite(output_path, grid)
    print(f"[OK] Imagem com transformações salva em: {output_path}")


if __name__ == "__main__":
    image_path = "./data/PAD-UFES-20/images/PAT_8_15_820.png"
    output_path = "./data/PAD-UFES-20/augmented_visual/grid_transforms_PAT_8_15_820.png"

    save_transforms_side_by_side(
        image_path=image_path,
        output_path=output_path,
        size=(224, 224),
    )