import cv2
import numpy as np

class ImagePointSelector:
    def __init__(self, image_paths, show_point_indices=False):
        self.image_paths = image_paths
        self.images = [cv2.imread(path) for path in image_paths]
        self.copies = [img.copy() for img in self.images]
        self.points = [[] for _ in image_paths]
        self.show_point_indices = show_point_indices
        self.last_image_clicked = 0  # Index of the last image clicked
        self.prepare_images()

    def pad_images_to_same_height(self):
        max_height = max(img.shape[0] for img in self.images)
        self.top_paddings = []
        for i, img in enumerate(self.images):
            diff = max_height - img.shape[0]
            self.images[i] = cv2.copyMakeBorder(img, diff // 2, diff - diff // 2, 0, 0, cv2.BORDER_CONSTANT)
            self.top_paddings.append(diff // 2)

    def prepare_images(self):
        self.pad_images_to_same_height()
        self.composite_image = np.hstack(self.images)
        self.original_composite_image = self.composite_image.copy()
        self.widths = [img.shape[1] for img in self.images]

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            scaled_x, scaled_y = x, y
            total_width = 0
            for idx, width in enumerate(self.widths):
                if scaled_x < total_width + width:
                    self.points[idx].append((scaled_x - total_width, scaled_y - self.top_paddings[idx]))
                    self.last_image_clicked = idx
                    print(f"Point added to Image {idx + 1}: ({scaled_x - total_width}, {scaled_y - self.top_paddings[idx]})")
                    break
                total_width += width
            self.draw_points()

    def remove_last_point(self):
        if self.points[self.last_image_clicked]:
            self.points[self.last_image_clicked].pop()
            print(f"Last point removed from Image {self.last_image_clicked + 1}")
            self.draw_points()

    def draw_points(self):
        self.composite_image = self.original_composite_image.copy()
        total_width = 0
        for idx, points in enumerate(self.points):
            for pidx, point in enumerate(points):
                cv2.circle(self.composite_image, (point[0] + total_width, point[1] + self.top_paddings[idx]), 5, (0, 255, 0), -1)
                if self.show_point_indices:
                    cv2.putText(self.composite_image, str(pidx), (point[0] + total_width + 10, point[1] + self.top_paddings[idx] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            total_width += self.widths[idx]
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
        return self.points



if __name__ == "__main__":
    ##### test
    # Example usage
    img_f1 = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/halloween_b2_v1/halloween_b2_v1_recon/images/00000.png"
    img_f2 = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/halloween_b2_v1/trig_eimgs/0000.png"

    selector = ImagePointSelector([img_f1, img_f2], show_point_indices=True)
    points_image1, points_image2 = selector.select_points()

    print("Points on Image 1:", points_image1)
    print("Points on Image 2:", points_image2)