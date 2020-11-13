from PIL import Image, ImageOps, ImageDraw

with open("./data/train.txt", "r") as train_f:
    lines = train_f.readlines()
    for line in lines:
        elemset = line.replace('\n','').split(" ")
        filename = elemset[0]
        image = Image.open(filename)
        draw = ImageDraw.Draw(image)
        filename_target = filename.replace("data/image_split", "target")
        print(filename_target)
        for polygon in elemset[1:]:
            pos_set = polygon.split(",")
            left = int(pos_set[0])
            top = int(pos_set[1])
            right = int(pos_set[2])
            bottle = int(pos_set[3])
            draw.line((left,top,right,bottle), fill=128)
        image.save(filename_target)

