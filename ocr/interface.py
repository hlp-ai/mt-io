

class OCR:

    def detect(self, image):
        """

        Args:
            image: image file path or image ndarray with shape (h, w, c) and value [0, 255]

        Returns:
            text list
        """
        raise NotImplementedError
