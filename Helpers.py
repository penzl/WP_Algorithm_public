import time
import os
import gzip
import _pickle as cpickle
from dataclasses import dataclass
import matplotlib.pyplot as plt
from private_Helpers import *

def gaussian_2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


def create_coordinates_on_source_and_target_planes(dim, a, z):
    points = np.indices((dim, dim)).T.reshape(-1, 2) * a / (dim - 1)
    z_values = np.zeros((len(points), 1), dtype=float)
    points_source = np.append(points, z_values, axis=1)  # points on left grid
    points_target = np.append(points, z_values + z, axis=1)  # points on right grid
    return points_source, points_target


def list_of_symmetric_elements(dim):
    array = np.reshape([x for x in range(0, dim * dim)], (dim, dim))
    return array[0:int(dim / 2), 0:int(dim / 2)].flatten()


def common_elements(list1, list2):
    return [element for element in list1 if element in list2]


def force_aspect(ax, aspect=1):
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * aspect)


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d h, %02d min, %02d s" % (hour, minutes, seconds)


def check_symmetries(A):
    return [np.all(A == np.flip(A)), np.all(A == np.flipud(A)),
            np.all(A == np.fliplr(A)), np.all(A == np.transpose(A))]


@dataclass
class AlgorithmInput:
    dim: int
    z: float
    a: int
    lmbd: float
    Beam: np.ndarray
    ImgTest: np.ndarray
    wish: np.ndarray
    AfterWPP: np.ndarray
    Mask: np.ndarray
    label: str
    n: int = 0
    meanafterWP: np.ndarray = np.array([])
    cmp: np.ndarray = np.array([])
    qr_diff: np.ndarray = np.array([])


class AlgorithmFigure:
    def __init__(self, data, dir_path):
        self.dir_path = dir_path
        self.data = data
        self.fig, self.axs = self.main_fig()

    def main_fig(self):
        a = self.data.a
        dim = self.data.dim
        z = self.data.z
        n = self.data.n
        label = self.data.label

        fig, axs = plt.subplots(3, 2, squeeze=True, num=1, figsize=(5.8, 9), clear=True)
        fig.subplots_adjust(top=1.2)
        fig.suptitle('Project1, dim={dim}, z={z}, n={n} {add_text}'
                     .format(dim=dim, z=z, n=n,
                             add_text=label), fontsize=12, y=0.994)

        axs[0, 0].imshow(abs(self.data.Beam) ** 2, extent=[0, a, 0, a], cmap='hot')
        axs[0, 0].set(title="Beam")

        axs[0, 1].imshow(abs(self.data.wish) ** 2, extent=[0, a, 0, a], cmap='hot')
        axs[0, 1].set(title="Target")

        axs[1, 0].imshow(np.mod(np.angle(self.data.AfterWPP) - np.pi, 2 * np.pi), extent=[0, a, 0, a], cmap='hot')
        axs[1, 0].set(title="Waveplate")

        axs[1, 1].imshow(abs(self.data.ImgTest) ** 2, extent=[0, a, 0, a], cmap='hot')
        axs[1, 1].set(title="Image")

        axs[2, 0].set_title("Mean Dev from Target", pad=15)
        axs[2, 0].plot(self.data.meanafterWP, 'k-')
        force_aspect(axs[2, 0], aspect=1)

        axs[2, 1].set_title("QR differences", pad=15)
        axs[2, 1].plot(self.data.qr_diff, 'k-')
        axs[2, 1].set_ylim([0, 50])
        force_aspect(axs[2, 1], aspect=1)

        plt.tight_layout()
        print("Plotting...")
        plt.pause(1)

        return fig, axs

    def update_fig(self, data):
        self.data = data
        self.fig.suptitle('Project1, dim={dim}, z={z}, n={n}, {add_text}'
                          .format(dim=self.data.dim, z=self.data.z, n=self.data.n,
                                  add_text=self.data.label), fontsize=12, y=0.994)

        self.axs[1, 0].imshow(np.mod(np.angle(self.data.AfterWPP) - np.pi, 2 * np.pi),
                              extent=[0, self.data.a, 0, self.data.a], cmap='hot')

        self.axs[1, 1].imshow(abs(self.data.ImgTest) ** 2,
                              extent=[0, self.data.a, 0, self.data.a], cmap='hot')
        self.axs[2, 0].clear()
        self.axs[2, 0].set_title("Mean Dev from Target", pad=15)
        self.axs[2, 0].plot(self.data.meanafterWP, 'k-')
        force_aspect(self.axs[2, 0], 1)
        f_name = self.dir_path + \
                 "/Project1_dim_{dim}_z_{z}_n_{n}" \
                     .format(dim=self.data.dim, z=self.data.z, n=self.data.n)
        self.fig.savefig(f_name + '.png')


class Algorithm:
    def __init__(self, data, dir_path, **kwargs):
        self.MultiCore = kwargs.get('MultiCore', False)
        self.Symmetrize = kwargs.get('Symmetrize', False)
        self.QR_detector_switch = kwargs.get('QR_detector_switch', False)
        self.numba = kwargs.get('numba', False)
        self.RemovePrevious = kwargs.get('RemovePrevious', False)
        self.AddToFolder = kwargs.get('AddToFolder', True)
        self.n_max = kwargs.get('n_max', 500)
        self.output_func = kwargs.get('output_func', AlgorithmFigure(data,dir_path).update_fig)
        self.data = data
        self.dir_path = dir_path

    def run_algorithm(self):
        points_source, points_target = \
            create_coordinates_on_source_and_target_planes(self.data.dim, self.data.a, self.data.z)

        print("\n ------------------------------------------")
        # self.data.n = self.data.n + 1
        Img = self.data.wish * np.exp(1j * np.angle(self.data.ImgTest))
        while self.data.n <= self.n_max:
            print("Calculating iteration:" + str(self.data.n) + "\n")
            t0 = time.time()
            print("Propagating one way...")
            self.data.AfterWPP = prop_fk_integral_v2(self.data.Mask, points_source, points_target, Img,
                                                     -self.data.lmbd,
                                                     self.data.dim, self.data.z,
                                                     Symmetrize=self.Symmetrize, MultiCore=self.MultiCore,
                                                     numba=self.numba)
            self.data.AfterWP = self.data.Beam * np.exp(1j * np.angle(self.data.AfterWPP))
            print("Propagating other way...")
            self.data.ImgTest = prop_fk_integral_v2(np.ones((self.data.dim, self.data.dim)), points_source,
                                                    points_target,
                                                    self.data.AfterWP, self.data.lmbd, self.data.dim, self.data.z,
                                                    Symmetrize=self.Symmetrize, MultiCore=self.MultiCore,
                                                    numba=self.numba)
            Img = self.data.wish * np.exp(1j * np.angle(self.data.ImgTest))

            self.data.cmp = np.append(self.data.cmp, np.sum(abs(self.data.ImgTest) ** 2 - abs(self.data.wish) ** 2))
            one = np.abs(self.data.ImgTest) / np.linalg.norm(np.abs(self.data.ImgTest))
            two = np.abs(self.data.wish) / np.linalg.norm(np.abs(self.data.wish))
            D = np.abs(one - two) ** 2
            self.data.meanafterWP = np.append(self.data.meanafterWP,
                                              np.sqrt(np.sum(D) / (len(one.flatten()) - 1)))
            self.output_func(self.data)
            # if QR_detector_switch:
            #     _, _, _, diff, threshold = main_qr_search(abs(ImgTest) ** 2, abs(wish) ** 2,
            #                                               data["qr_size"] * data["qr_target_ratio"])
            #     qr_diff = np.append(data["qr_diff"], diff)
            #     thresholds = np.append(data["thresholds"], threshold)
            #     qr_diff0 = copy.deepcopy(qr_diff)
            #     qr_diff0[qr_diff0 > 30] = np.nan
            #     axs[2, 1].set_title("QR differences", pad=15)
            #     axs[2, 1].plot(qr_diff0, 'k-')
            #     axs[2, 1].set_ylim([0, 50])
            #     forceAspect(axs[2, 1], 1)
            #     data["qr_diff"] = qr_diff
            #     data["thresholds"] = thresholds

            f_name = self.dir_path + \
                     "/Project1_dim_{dim}_z_{z}_n_{n}" \
                         .format(dim=self.data.dim, z=self.data.z, n=self.data.n)
            with gzip.GzipFile(f_name + ".pgz", 'wb') as f:
                cpickle.dump(
                    {
                        "n": self.data.n,
                        "ImgTest": self.data.ImgTest,
                        "meanafterWP": self.data.meanafterWP,
                        "cmp": self.data.cmp,
                        "qr_diff": self.data.qr_diff,
                    }, f)
            print("Saved temp!")

            # add_to_zip([f_name + '.png', f_name + ".dat"],
            #           dir_path + "/Project1_dim_{dim}_z_{z}_{add_text}.zip".format(dim=dim, z=z,
            #                                                                        add_text=data["add_text"]))
            # print("Added to zip...")
            time.sleep(1)
            self.folder_stuff(self.dir_path, self.data.dim, self.data.z, self.data.n)
            t1 = time.time()
            print("Time: " + convert(t1 - t0))

            print("------------------------------------------")

            plt.pause(1)
            self.data.n = self.data.n + 1
        else:
            print("Aborted")

    def check_symmetries(self):
        img = self.data.wish * np.exp(1j * np.angle(self.data.ImgTest))
        print('Checking symmetries of data...')
        print('        | flip | flipud | fliplr | transpose |')
        print('Beam  = ' + str(check_symmetries(self.data.Beam)))
        print('Target= ' + str(check_symmetries(self.data.wish)))
        print('Img   = ' + str(check_symmetries(img)))
        print('Mask  = ' + str(check_symmetries(self.data.Mask)))

    def folder_stuff(self, dir_path, dim, z, n):
        if self.RemovePrevious:
            try:
                os.remove(dir_path + \
                          "/Project1_dim_{dim}_z_{z}_n_{n}.pgz".format
                          (dim=dim, z=z, n=n - 1))
            except FileNotFoundError:
                pass
            try:
                os.remove(self.dir_path + \
                          "/Project1_dim_{dim}_z_{z}_n_{n}.png".format
                          (dim=dim, z=z, n=n - 1))
            except FileNotFoundError:
                pass
        if self.AddToFolder:
            try:
                os.mkdir(self.dir_path + "/Other_Its")
            except FileExistsError:
                # print('Directory already exists...')
                pass
            try:
                os.remove(self.dir_path
                          + "/Other_Its/Project1_dim_{dim}_z_{z}_n_{n}.pgz".format
                          (dim=dim, z=z, n=n - 1)
                          )
            except FileNotFoundError:
                pass
            try:
                os.remove(self.dir_path
                          + "/Other_Its/Project1_dim_{dim}_z_{z}_n_{n}.png".format
                          (dim=dim, z=z, n=n - 1)
                          )
            except FileNotFoundError:
                pass
            try:
                os.rename(self.dir_path
                          + "/Project1_dim_{dim}_z_{z}_n_{n}.pgz".format(dim=dim, z=z, n=n - 1),
                          dir_path
                          + "/Other_Its/Project1_dim_{dim}_z_{z}_n_{n}.pgz".format(dim=dim, z=z, n=n - 1)
                          )
                os.rename(dir_path
                          + "/Project1_dim_{dim}_z_{z}_n_{n}.png".format(dim=dim, z=z, n=n - 1),
                          dir_path
                          + "/Other_Its/Project1_dim_{dim}_z_{z}_n_{n}.png".format(dim=dim, z=z, n=n - 1)
                          )
            except FileNotFoundError:
                pass
