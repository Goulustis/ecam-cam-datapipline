import cv2
import numpy as np

class ImagePointSelector:
    def __init__(self, image_path1, image_path2, show_point_indices=False):
        self.image_path1, self.image_path2 = image_path1, image_path2
        self.original_image1 = cv2.imread(image_path1)
        self.original_image2 = cv2.imread(image_path2)
        self.image1 = self.original_image1.copy()
        self.image2 = self.original_image2.copy()
        self.points1 = []
        self.points2 = []
        self.show_point_indices = show_point_indices
        self.last_image_clicked = 1  # 1 for image1, 2 for image2
        self.prepare_images()

    def pad_image_to_same_height(self, img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 > h2:
            diff = h1 - h2
            img2 = cv2.copyMakeBorder(img2, diff // 2, diff - diff // 2, 0, 0, cv2.BORDER_CONSTANT)
        elif h2 > h1:
            diff = h2 - h1
            img1 = cv2.copyMakeBorder(img1, diff // 2, diff - diff // 2, 0, 0, cv2.BORDER_CONSTANT)
        
        return img1, img2

    def prepare_images(self):
        self.image1, self.image2 = self.pad_image_to_same_height(self.image1, self.image2)
        self.composite_image = np.hstack((self.image1, self.image2))
        self.original_composite_image = self.composite_image.copy()
        self.half_width = self.composite_image.shape[1] // 2
        self.top_padding1 = (self.image1.shape[0] - self.original_image1.shape[0]) // 2
        self.top_padding2 = (self.image2.shape[0] - self.original_image2.shape[0]) // 2

    def click_event(self, event, x, y, flags, params):
        # Calculate scale factors
        scale_x = self.original_composite_image.shape[1] / self.composite_image.shape[1]
        scale_y = self.original_composite_image.shape[0] / self.composite_image.shape[0]
        scaled_x, scaled_y = int(x * scale_x), int(y * scale_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if scaled_x < self.original_composite_image.shape[1] // 2:
                self.points1.append((scaled_x, scaled_y - self.top_padding1))
                self.last_image_clicked = 1
                print(f"Point added to Image 1: ({scaled_x}, {scaled_y - self.top_padding1})")
            else:
                self.points2.append((scaled_x - self.half_width, scaled_y - self.top_padding2))
                self.last_image_clicked = 2
                print(f"Point added to Image 2: ({scaled_x - self.half_width}, {scaled_y - self.top_padding2})")
            self.draw_points()

    def remove_last_point(self):
        if self.last_image_clicked == 1 and self.points1:
            self.points1.pop()
            print("Last point removed from Image 1")
        elif self.last_image_clicked == 2 and self.points2:
            self.points2.pop()
            print("Last point removed from Image 2")
        self.draw_points()

    def draw_points(self):
        self.composite_image = self.original_composite_image.copy()
        for idx, point in enumerate(self.points1):
            cv2.circle(self.composite_image, (point[0], point[1] + self.top_padding1), 5, (0, 0, 255), -1)
            if self.show_point_indices:
                cv2.putText(self.composite_image, str(idx), (point[0] + 10, point[1] + self.top_padding1 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        for idx, point in enumerate(self.points2):
            cv2.circle(self.composite_image, (point[0] + self.half_width, point[1] + self.top_padding2), 5, (0, 255, 0), -1)
            if self.show_point_indices:
                cv2.putText(self.composite_image, str(idx), (point[0] + self.half_width + 10, point[1] + self.top_padding2 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Composite Image', self.composite_image)

    def select_points(self):
        cv2.namedWindow('Composite Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Composite Image', self.original_composite_image.shape[1], self.original_composite_image.shape[0])
        cv2.setMouseCallback('Composite Image', self.click_event)

        while True:
            self.draw_points()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.remove_last_point()

        cv2.destroyAllWindows()
        return self.points1, self.points2


if __name__ == "__main__":
    ##### test
    # Example usage
    img_f1 = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/halloween_b2_v1/halloween_b2_v1_recon/images/00000.png"
    img_f2 = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/halloween_b2_v1/trig_eimgs/0000.png"

    selector = ImagePointSelector(img_f1, img_f2, show_point_indices=True)
    points_image1, points_image2 = selector.select_points()

    print("Points on Image 1:", points_image1)
    print("Points on Image 2:", points_image2)