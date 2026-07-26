from PIL import Image, ImageDraw, ImageFont

def add_title_to_image(image, title):
    # Converte a imagem para RGB, caso não esteja nesse formato
    image = image.convert("RGB")
    
    # Cria um objeto ImageDraw para desenhar na imagem
    draw = ImageDraw.Draw(image)
    
    # Define a fonte e o tamanho (tentando usar a "arial.ttf", se não, utiliza uma fonte padrão)
    try:
        font = ImageFont.truetype("arial.ttf", 30)  # Ou outra fonte disponível no seu sistema
    except IOError:
        font = ImageFont.load_default()
    
    # Pega as dimensões da imagem
    width, height = image.size
    
    # Define a posição do título (centralizado no topo)
    text_width, text_height = draw.textsize(title, font)
    position = ((width - text_width) // 2, 10)  # Centraliza o texto na parte superior
    
    # Adiciona o título à imagem
    draw.text(position, title, font=font, fill="white")
    
    return image

def create_gif(image_paths, output_gif_path, titles, duration=500):
    images = []
    
    for i, image_path in enumerate(image_paths):
        # Abre a imagem
        image = Image.open(image_path)
        
        # Verifica se há um título correspondente e o adiciona à imagem
        title = titles[i] if i < len(titles) else "Imagem " + str(i + 1)  # Título padrão se não houver título suficiente
        image_with_title = add_title_to_image(image, title)
        
        # Adiciona a imagem com título à lista de imagens
        images.append(image_with_title)
    
    # Cria o GIF a partir das imagens
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 0 significa loop infinito
    )

if __name__ == "__main__":
    # Lista de caminhos das imagens
    # image_paths = [
    #     "Heatmaps-unfrozen-last-layer-weights.png", 
    #     "Heatmaps-unfrozen-last-layer-weights-missing-metadata.png", 
    #     "Heatmaps-unfrozen-last-layer-weights-missing-metadata-with-age.png",
    #     "Heatmaps-unfrozen-last-layer-weights-missing-metadata-with-bleed.png", 
    #     "Heatmaps-unfrozen-last-layer-weights-missing-metadata-with-changed.png",
    #     "Heatmaps-unfrozen-last-layer-weights-missing-metadata-with-elevation.png", 
    #     "Heatmaps-unfrozen-last-layer-weights-missing-metadata-with-grew.png"
    # ]
    # image_paths = [
    #     "Heatmaps-frozen-weights.png", 
    #     "Heatmaps-frozen-weights-missing-metadata.png", 
    #     "Heatmaps-frozen-weights-missing-metadata-with-age.png",
    #     "Heatmaps-frozen-weights-missing-metadata-with-bleed.png", 
    #     "Heatmaps-frozen-weights-missing-metadata-with-changed.png",
    #     "Heatmaps-frozen-weights-missing-metadata-with-elevation.png", 
    #     "Heatmaps-frozen-weights-missing-metadata-with-grew.png"
    # ]
    image_paths = [
        "Heatmaps-unfrozen-weights.png", 
        "Heatmaps-unfrozen-weights-missing-metadata.png", 
        "Heatmaps-unfrozen-weights-missing-metadata-with-age.png",
        "Heatmaps-unfrozen-weights-missing-metadata-with-bleed.png", 
        "Heatmaps-unfrozen-weights-missing-metadata-with-changed.png",
        "Heatmaps-unfrozen-weights-missing-metadata-with-elevation.png", 
        "Heatmaps-unfrozen-weights-missing-metadata-with-grew.png"
    ]
    # Títulos para cada imagem
    titles = [
        "Layer Weights (Unfrozen)", 
        "Missing Metadata", 
        "Metadata with Age", 
        "Metadata with Bleed", 
        "Metadata with Changes", 
        "Metadata with Elevation", 
        "Metadata with Growth"
    ]
    
    # Caminho de saída do GIF
    output_gif_path = "Heatmaps-unfrozen-weights.gif"
    
    # Criação do GIF
    create_gif(image_paths, output_gif_path, titles)
