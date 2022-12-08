from pycocotools.cocoeval import Params
class CustomizedEvalParams(Params):
    def __init__(self, coco_dataset):
        self.smallObjAreaRng, self.mediumObjAreaRng, _ = self.__calc_object_sizes(coco_dataset) # large obj limit is not needed
        super().__init__()

    def setDetParams(self):
        super().setDetParams()
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, self.smallObjAreaRng], [self.smallObjAreaRng, self.mediumObjAreaRng], [self.mediumObjAreaRng, 1e5 ** 2]]

    def __calc_object_sizes(self, coco):
        annotation_areas = []
        for i in coco.anns:
            annotation_areas.append(coco.anns[i]['area'])

        annotation_areas.sort()

        small = annotation_areas[len(annotation_areas) // 3 - 1] + 1
        medium = annotation_areas[len(annotation_areas) * 2 // 3 - 1] + 1
        large = annotation_areas[len(annotation_areas) - 1] + 1

        print(f"Calculated object sizes from distribution \n Small limit: {small} \n Medium limit: {medium} \n Largest object area: {large}")

        # import matplotlib.pyplot as plt
        # plt.hist(annotation_areas, edgecolor="green", bins=50)
        # plt.show()
        # plt.hist(annotation_areas, edgecolor="green", bins=[0, small, medium, large])
        # plt.show()

        return small, medium, large