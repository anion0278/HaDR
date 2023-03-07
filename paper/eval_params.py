from pycocotools.cocoeval import Params

def show_hist(data, bins, y_lim = None, enable_bar_labels = True):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 2.5))
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

        colors = []
        for c, p in zip(col, bars):
            plt.setp(p, 'facecolor', cm(c))
            colors.append(cm(c))

        if enable_bar_labels:
            from matplotlib.patches import Rectangle
            handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in colors]
            labels= ["small","medium", "large"]
            plt.legend(handles, labels, loc="lower right")

        plt.xlabel('Instance area')
        plt.ylabel('Number of instances')
        if y_lim is not None: 
            ax = plt.gca()
            ax.set_ylim([0, y_lim])
        plt.subplots_adjust(left = 0.17, bottom=0.2, right=0.95)
        plt.show(block=True)

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
        # show_hist(annotation_areas,[0, 32 ** 2, 96 ** 2, large], y_lim=690.0)
        # show_hist(annotation_areas,[0, small, medium, large], y_lim = 470.0)
        # show_hist(annotation_areas,50, enable_bar_labels = False)

        return small, medium, large

    