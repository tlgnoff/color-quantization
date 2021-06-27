import sys
import kmeans
import numpy as np
from PIL import Image

def main():
    # reading pixels
    print("Reading pixels...", end="\t")
    
    # read image pixels
    # and have a list in 1 X (width * height) dimensions

    image_path = sys.argv[1]
    number_of_colors = int(sys.argv[2])
    max_iterations = int(sys.argv[3])
    epsilon = float(sys.argv[4])

    image = Image.open(image_path)
    inp = Image.Image.getdata(image)

    print("DONE")
    
    model = kmeans.KMeans(
        X=np.array(inp),
        n_clusters=number_of_colors,
        max_iterations=max_iterations,
        epsilon=epsilon,
        distance_metric="euclidian"
    )
    print("Fitting...")
    model.fit()    
    print("Fitting... DONE")

    print("Predicting...")
    color1 = (134, 66, 176)
    color2 = (34, 36, 255)
    color3 = (94, 166, 126)
    print(f"Prediction for {color1} is cluster {model.predict(color1)}")
    print(f"Prediction for {color2} is cluster {model.predict(color2)}")
    print(f"Prediction for {color3} is cluster {model.predict(color3)}")

    # replace image pixels with color palette
    # (cluster centers) found in the model
    width, height = image.size
    for x in range(width):
        for y in range(height):
            old_pixel = image.getpixel((x, y))
            new_pixel = model.cluster_centers[model.predict(old_pixel)]
            new_pixel = (int(new_pixel[0]), int(new_pixel[1]), int(new_pixel[2]))
            image.putpixel((x, y), new_pixel)

    # save the final image
    filename = "{}_{}_colors_{}_epochs_epsilon_{}.png".format(image_path, number_of_colors, max_iterations, epsilon)
    image.save(filename)

if __name__ == "__main__":
    main()