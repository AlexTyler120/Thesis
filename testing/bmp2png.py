from PIL import Image
import cv2
def bmp_to_png(bmp_path, png_path):
    print(bmp_path)
    try:
        # Open the BMP image
        with Image.open(bmp_path) as bmp_image:
            # Convert and save as PNG
            bmp_image.save(png_path, "PNG")
        print("Conversion successful!")
    except Exception as e:
        print(f"Error converting BMP to PNG: {e}")

def main():
    # images = [10,20,30,40,50,60,70,80,90,100]
    # objects = ["apple", "Lorikeet"]

    path = r"C:/Users/Alex/OneDrive - The University of Sydney (Students)/UNI/Thesis/testing/pointlight/furtheraway"

    bmp_path = f"{path}/nopol.bmp"
    png_path = f"{path}/png/nopol.png"
    bmp_to_png(bmp_path, png_path)


if __name__ == "__main__":
    main()