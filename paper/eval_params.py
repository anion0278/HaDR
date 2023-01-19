from pycocotools.cocoeval import Params

def show_hist(data, bins, enable_bar_labels = True):
        import matplotlib.pyplot as plt
        counts, edges, bars = plt.hist(data, edgecolor="black", bins=bins)
        if enable_bar_labels: plt.bar_label(bars)
        cm = plt.cm.get_cmap('Blues')

        # Plot histogram.
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # scale values to interval [0,1]
        col = bin_centers - min(bin_centers)
        col /= max(col)

        range_corr = 0.33
        col /= (1 / range_corr)
        col += range_corr

        for c, p in zip(col, bars):
            plt.setp(p, 'facecolor', cm(c))

        plt.xlabel('Instance area')
        plt.ylabel('Number of instances')
        plt.show()

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

        # thresholds for obj sizes are 1/3, 2/3 and 3/3
        small = annotation_areas[len(annotation_areas) // 3] 
        medium = annotation_areas[len(annotation_areas) * 2 // 3] 
        large = annotation_areas[len(annotation_areas) - 1] 

        print(f"Calculated object sizes from distribution \n Small limit: {small} \n Medium limit: {medium} \n Largest object area: {large}")

        # COCO: self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        show_hist(annotation_areas,[0, 32 ** 2, 96 ** 2, large])
        show_hist(annotation_areas,[0, small, medium, large])
        show_hist(annotation_areas,50, False)

        return small, medium, large

    