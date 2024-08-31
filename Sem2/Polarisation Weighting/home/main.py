import Images
import ShiftEstimate

def main():
    RESIZE_VAR = 0.3
    GREY = False

    img1, _, _ = Images.read_image("python/test_im/ball/ball_0.png", RESIZE_VAR, GREY)
    img2, _, _ = Images.read_image("python/test_im/ball/ball_90.png", RESIZE_VAR, GREY)
    transformed_image = Images.create_shifted_image(img1, img2, 12, True)
    shift_est = ShiftEstimate.compute_pixel_shift(transformed_image)
    print(f"Shift estimate: {shift_est}")

if __name__ == "__main__":
    main()