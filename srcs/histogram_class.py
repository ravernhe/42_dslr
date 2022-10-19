import matplotlib.pyplot as plt

class Histogram:
    def __init__(self, df, legend, xlabel, ylabel) -> None:
        self.df = df
        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel


    def show_histogram(self, course):
        
        x = self.df.loc["Gryffindor"][course]
        y = self.df.loc["Hufflepuff"][course]
        z = self.df.loc["Ravenclaw"][course]
        a = self.df.loc["Slytherin"][course]

        hist = plt.figure()
        plt.hist(x, histtype="stepfilled", align="mid", color="red", alpha=0.5, figure=hist)
        plt.hist(y, histtype="stepfilled", align="mid", color="yellow", alpha=0.5, figure=hist)
        plt.hist(z, histtype="stepfilled", align="mid", color="blue", alpha=0.5, figure=hist)
        plt.hist(a, histtype="stepfilled", align="mid", color="green", alpha=0.5, figure=hist)

        plt.title(course)
        hist.legend(self.legend, loc='upper right', frameon=False)
        plt.xlabel(self.xlabel, figure=hist)
        plt.ylabel(self.ylabel, figure=hist)

    def draw_histogram(self, course="all"):
        if course == "all":
            for name in self.df.select_dtypes("number"):
                self.show_histogram(name) 
        else:
           self.show_histogram(course) 
        plt.show()
