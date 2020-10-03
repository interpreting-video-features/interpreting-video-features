import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os


def vizualize_results(orig_seq, pert_seq, mask, rootDir=None, case="0", markImgs=True, iterTest=False):
    if rootDir is None:
        rootDir = "vizualisations/" + args.subDir + "/"
    # if(iterTest):
    rootDir += "/PerturbImgs/"

    if not os.path.exists(rootDir):
        os.makedirs(rootDir)

    origPy = orig_seq.cpu().detach().numpy()
    pertPy = pert_seq.cpu().detach().numpy()
    for i in range(orig_seq.shape[1]):

        if markImgs:
            origPy[1:, i, :10, :10] = 0
            origPy[0, i, :10, :10] = mask[i].detach().cpu() * 255
            pertPy[1:, i, :10, :10] = 0
            pertPy[0, i, :10, :10] = mask[i].detach().cpu() * 255
        # result = Image.fromarray(origPy[:,i,:,:].transpose(1,2,0).astype(np.uint8))
        # result.save(rootDir+"case"+case+"orig"+str(i)+".png")
        result = Image.fromarray(pertPy[:, i, :, :].transpose(1, 2, 0).astype(np.uint8))
        result.save(rootDir + "case" + case + "pert" + str(i) + ".png")
    f = open(rootDir + "case" + case + ".txt", "w+")
    f.write(str(mask.detach().cpu()))
    f.close()


def vizualize_results_on_gradcam(gradCamImage, mask, rootDir, case="0", roundUpMask=True, imageWidth=224,
                                 imageHeight=224):
    # print("gradCamType: ", gradCamImage.type)
    try:
        mask = mask.detach().cpu()
    except:
        print("mask was already on cpu")
    if not os.path.exists(rootDir):
        os.makedirs(rootDir)

    dots = find_temp_mask_red_dots(imageWidth, imageHeight, mask, roundUpMask)

    dotOffset = imageWidth * 2
    for i in range(len(mask)):
        for j, dot in enumerate(dots):

            if i == j:
                intensity = 255
            else:
                intensity = 150
            gradCamImage[:, i, dot["yStart"]:, dotOffset + dot["xStart"]:dotOffset + dot["xEnd"]] = 0
            gradCamImage[dot["channel"], i, dot["yStart"]:,
            dotOffset + dot["xStart"]:dotOffset + dot["xEnd"]] = intensity

            result = Image.fromarray(gradCamImage[::-1, i, :, :].transpose(1, 2, 0).astype(np.uint8), mode="RGB")
            result.save(rootDir + "/case" + case + "_" + str(i) + ".png")

    f = open(rootDir + "/MASKVALScase" + case + ".txt", "w+")
    f.write(str(mask.detach().cpu()))
    f.close()


def find_temp_mask_red_dots(imageWidth, imageHeight, mask, roundUpMask):
    maskLen = len(mask)
    dotWidth = int(imageWidth // (maskLen + 4))
    dotPadding = int((imageWidth - (dotWidth * maskLen)) // maskLen)
    dotHeight = int(imageHeight // 20)
    dots = []

    for i, m in enumerate(mask):

        if roundUpMask:
            if mask[i] > 0.5:
                mask[i] = 1
            else:
                mask[i] = 0

        dot = {'yStart': -dotHeight,
               'yEnd': imageHeight,
               'xStart': i * (dotWidth + dotPadding),
               'xEnd': i * (dotWidth + dotPadding) + dotWidth}
        if mask[i] == 0:
            dot['channel'] = 1
        else:
            dot['channel'] = 2

        dots.append(dot)

    return dots


def create_image_arrays(input_sequence, gradcamMask, timeMask, intraBidx, temporalMaskType, output_folder, targTag,
                        RESIZE_FLAG, RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT):
    input_to_model = input_sequence[intraBidx][None, :, :, :, :]

    input_data_unnormalised = input_to_model[0].cpu().permute(1, 2, 3, 0).numpy()
    input_data_unnormalised = np.flip(input_data_unnormalised, 3)

    combined_images = []
    for i in range(input_data_unnormalised.shape[0]):
        input_data_img = input_data_unnormalised[i, :, :, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * gradcamMask[i]), cv2.COLORMAP_JET)
        if RESIZE_FLAG:
            input_data_img = cv2.resize(input_data_img, (RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT))
            heatmap = cv2.resize(heatmap, (RESIZE_SIZE_WIDTH, RESIZE_SIZE_HEIGHT))
        heatmap = np.float32(heatmap)  # / 255
        cam = heatmap + np.float32(input_data_img)
        cam = cam / np.max(cam)

        combined_img = np.concatenate((np.uint8(input_data_img), np.uint8(255 * cam), \
                                       np.uint8(perturb_sequence(input_sequence, timeMask.clone(), \
                                                                 perturbation_type=temporalMaskType, snap_values=True)[intraBidx][
                                                :, i, :, :].permute(1, 2, 0).detach().cpu().numpy())[:, :, ::-1]) \
                                      , axis=1)  # had 255* on both
        combined_images.append(combined_img)
        cv2.imwrite(os.path.join(output_folder, "img%02d.jpg" % (i + 1)), combined_img)

    path_to_combined_gif = os.path.join(output_folder, "mygif.gif")
    os.system("convert -delay 10 -loop 0 {}.jpg {}".format(
        os.path.join(output_folder, "*"),
        path_to_combined_gif))

    combined_images = np.transpose(np.array(combined_images), (3, 0, 1, 2))
    vizualize_results_on_gradcam(combined_images, timeMask, rootDir=output_folder, case=temporalMaskType + targTag)

    return combined_images


class PlotLearning(object):
    def __init__(self, save_path, num_classes):
        self.accuracy = []
        self.val_accuracy = []
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.save_path_loss = os.path.join(save_path, 'loss_plot.png')
        self.save_path_accu = os.path.join(save_path, 'accu_plot.png')
        self.save_path_lr = os.path.join(save_path, 'lr_plot.png')
        self.init_loss = -np.log(1.0 / num_classes)

    def plot(self, logs):
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.learning_rates.append(logs.get('learning_rate'))

        best_val_acc = max(self.val_accuracy)
        best_train_acc = max(self.accuracy)
        best_val_epoch = self.val_accuracy.index(best_val_acc)
        best_train_epoch = self.accuracy.index(best_train_acc)

        plt.figure(1)
        plt.gca().cla()
        plt.ylim(0, 1)
        plt.plot(self.accuracy, label='train')
        plt.plot(self.val_accuracy, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_acc, best_train_epoch, best_train_acc))
        plt.legend()
        plt.savefig(self.save_path_accu)

        best_val_loss = min(self.val_losses)
        best_train_loss = min(self.losses)
        best_val_epoch = self.val_losses.index(best_val_loss)
        best_train_epoch = self.losses.index(best_train_loss)

        plt.figure(2)
        plt.gca().cla()
        plt.ylim(0, self.init_loss)
        plt.plot(self.losses, label='train')
        plt.plot(self.val_losses, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_loss, best_train_epoch, best_train_loss))
        plt.legend()
        plt.savefig(self.save_path_loss)

        min_learning_rate = min(self.learning_rates)
        max_learning_rate = max(self.learning_rates)

        plt.figure(2)
        plt.gca().cla()
        plt.ylim(0, max_learning_rate)
        plt.plot(self.learning_rates)
        plt.title("max_learning_rate-{0:.6f}, min_learning_rate-{1:.6f}".format(max_learning_rate, min_learning_rate))
        plt.savefig(self.save_path_lr)


