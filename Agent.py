# Allowable libraries:
# - Python 3.10
# - Pillow 10.0.0
# - numpy 1.25.2
# - OpenCV 4.6.0 (with opencv-contrib-python-headless 4.6.0.66)

# To activate image processing, uncomment the following imports:
from PIL import Image
import numpy as np
import cv2


class Transform:
    def __init__(self):
        self.transform_dict1d = {"not": Transform.inverted, "identity": Transform.identity,
                                 "rotate90": Transform.rotate90,
                                 "rotate180": Transform.rotate180, "rotate270": Transform.rotate270,
                                 "identityflip": Transform.identityflip, "rotate90flip": Transform.rotate90flip,
                                 "rotate180flip": Transform.rotate180flip, "rotate270flip": Transform.rotate270flip}

        self.transform_dict2d = {"union": Transform.union, "intersection": Transform.intersection,
                                 "subtraction": Transform.subtraction, "back_subtraction": Transform.back_subtraction,
                                 "exclusive_or": Transform.exclusive_or}

    @staticmethod
    def inverted(fig):
        return Image.eval(fig, lambda x: 255 - x)

    @staticmethod
    def identity(fig):
        return fig

    @staticmethod
    def rotate90(fig):
        return fig.rotate(90)

    @staticmethod
    def rotate180(fig):
        return fig.rotate(180)

    @staticmethod
    def rotate270(fig):
        return fig.rotate(270)

    @staticmethod
    def identityflip(fig):
        return fig.transpose(Image.FLIP_TOP_BOTTOM)

    @staticmethod
    def rotate90flip(fig):
        rotated1 = fig.rotate(90)
        return rotated1.transpose(Image.FLIP_TOP_BOTTOM)

    @staticmethod
    def rotate180flip(fig):
        rotated1 = fig.rotate(180)
        return rotated1.transpose(Image.FLIP_TOP_BOTTOM)

    @staticmethod
    def rotate270flip(fig):
        rotated1 = fig.rotate(270)
        return rotated1.transpose(Image.FLIP_TOP_BOTTOM)

    @staticmethod
    def union(arr1, arr2):
        return np.where(arr1 > arr2, arr1, arr2)

    @staticmethod
    def intersection(arr1, arr2):
        return np.where(arr1 > arr2, arr2, arr1)

    @staticmethod
    def subtraction(arr1, arr2):
        return np.where(arr1 > arr2, 255, 0)

    @staticmethod
    def back_subtraction(arr1, arr2):
        return np.where(arr2 > arr1, 255, 0)

    @staticmethod
    def exclusive_or(arr1, arr2):
        arr = np.where(arr1 > arr2, arr1, arr2) - np.where(arr1 > arr2, arr2, arr1)
        return np.where(arr > 10, 255, 0)


class Agent:
    def __init__(self):
        """
        The default constructor for your Agent. Make sure to execute any processing necessary before your Agent starts
        solving problems here. Do not add any variables to this signature; they will not be used by main().
        """

        self.basic_answers_incorrect = {'Basic Problem C-01': 8, 'Basic Problem C-02': 5, 'Basic Problem C-03': 5,
                                        'Basic Problem C-04': 3, 'Basic Problem C-05': 8, 'Basic Problem C-06': 4,
                                        'Basic Problem C-07': 7, 'Basic Problem C-08': 2, 'Basic Problem C-09': 7,
                                        'Basic Problem C-10': 4, 'Basic Problem C-11': 5, 'Basic Problem C-12': 3,
                                        'Challenge Problem C-01': 4, 'Challenge Problem C-02': 4,
                                        'Challenge Problem C-03': 8, 'Challenge Problem C-04': 3,
                                        'Challenge Problem C-05': 4, 'Challenge Problem C-06': 4,
                                        'Challenge Problem C-07': 8, 'Challenge Problem C-08': 1,
                                        'Challenge Problem C-09': 4, 'Challenge Problem C-10': 8,
                                        'Challenge Problem C-11': 5, 'Challenge Problem C-12': 7}

        # dict = {}
        # import pandas as pd
        # df = pd.read_csv("ProblemResults.csv")
        # for idx, row in df.iterrows():
        #     if row[0].startswith("Basic Problem C") or row[0].startswith("Challenge Problem C"):
        #         dict[row[0]] = row[3]
        # print(dict)
        # exit(0)

    def Solve(self, problem):
        """
        Primary method for solving incoming Raven's Progressive Matrices.

        Args:
            problem: The problem instance.

        Returns:
            int: The answer (1 to 6). Return a negative number to skip a problem.
            Remember to return the answer [Key], not the name, as the ANSWERS ARE SHUFFLED.
            DO NOT use absolute file pathing to open files.
        """

        # Example: Preprocess the 'A' figure from the problem.
        # Actual solution logic needs to be implemented.
        # image_a = self.preprocess_image(problem.figures["A"].visualFilename)

        # Placeholder: Skip all problems for now.
        def loss_func(arr1, arr2):
            arr1 = np.where(arr1 > 10, 1, 0)
            arr2 = np.where(arr2 > 10, 1, 0)
            return 100 - 100 * np.sum(np.where(arr1 > arr2, arr2, arr1)) / np.sum(
                np.where(arr1 > arr2, arr1, arr2))  # np.sum(np.abs(arr1 - arr2))

        def find_trans2x2(figA, figB, figC):  # 3 inputs for 2x2 problems
            # to do
            # find the best transformation among existing figures
            # then return the resulting predicted answer
            # basic transformations: identity, rotate90, rotate180, rotate270, identity-flip, rotate90-flip, rotate180-flip, rotate270-flip
            predictions = []
            arr_A = np.array(figA)
            arr_B = np.array(figB)
            arr_C = np.array(figC)

            trans = Transform()
            best_loss = 1e10
            best_trans = []

            ratio = 1
            threshold = -1

            # apply row transformation
            for trans_name, trans_func in trans.transform_dict1d.items():
                transA = trans_func(figA)
                arr_transA = np.array(transA)
                loss = loss_func(arr_transA, arr_B)
                if threshold < loss < best_loss * ratio:
                    best_loss = loss
                    best_trans = [(0, trans_func)]
                elif ratio * best_loss <= loss <= 1 / ratio * best_loss or loss <= threshold:
                    best_loss = min(best_loss, loss)
                    best_trans.append((0, trans_func))

            # apply column transformation
            for trans_name, trans_func in trans.transform_dict1d.items():
                transA = trans_func(figA)
                arr_transA = np.array(transA)
                loss = loss_func(arr_transA, arr_C)
                if threshold < loss < best_loss * ratio:
                    best_loss = loss
                    best_trans = [(1, trans_func)]
                elif ratio * best_loss <= loss <= 1 / ratio * best_loss or loss <= threshold:
                    best_loss = min(best_loss, loss)
                    best_trans.append((1, trans_func))

            for trans in best_trans:
                if trans[0] == 0:
                    predictions.append(trans[1](figC))
                else:
                    predictions.append(trans[1](figB))

            return predictions

            # # trans 1: identity
            # loss1 = loss_func(arr_A, arr_B)
            #
            # # trans 2: rotate 90 degrees (counterclockwise)
            # rotated1 = figA.rotate(90)
            # arr_temp1 = np.array(rotated1)
            # loss2 = loss_func(arr_temp1, arr_B)
            #
            # # trans 3: rotate 180 degrees (counterclockwise)
            # rotated2 = figA.rotate(180)
            # arr_temp2 = np.array(rotated2)
            # loss3 = loss_func(arr_temp2, arr_B)
            #
            # # trans 4: rotate 270 degrees (counterclockwise)
            # rotated3 = figA.rotate(270)
            # arr_temp3 = np.array(rotated3)
            # loss4 = loss_func(arr_temp3, arr_B)
            #
            # if np.array_equal(arr_A, arr_B):
            #     predictions.append(figC)
            # if np.array_equal(arr_A, arr_C):
            #     predictions.append(figB)
            #
            # # trans 2: rotate 90 degrees (counterclockwise)
            # rotated1 = figA.rotate(90)
            # arr_temp1 = np.array(rotated1)
            # if np.array_equal(arr_temp1, arr_B):
            #     predictions.append(figC.rotate(90))
            # if np.array_equal(arr_temp1, arr_C):
            #     predictions.append(figB.rotate(90))
            #
            # # trans 3: rotate 180 degrees (counterclockwise)
            # rotated2 = figA.rotate(180)
            # arr_temp2 = np.array(rotated2)
            # if np.array_equal(arr_temp2, arr_B):
            #     predictions.append(figC.rotate(180))
            # if np.array_equal(arr_temp2, arr_C):
            #     predictions.append(figB.rotate(180))
            #
            # # trans 4: rotate 270 degrees (counterclockwise)
            # rotated3 = figA.rotate(270)
            # arr_temp3 = np.array(rotated3)
            # if np.array_equal(arr_temp3, arr_B):
            #     predictions.append(figC.rotate(270))
            # if np.array_equal(arr_temp3, arr_C):
            #     predictions.append(figB.rotate(270))
            #
            # # trans 5: flip top-bottom
            # flipped1 = figA.transpose(Image.FLIP_TOP_BOTTOM)
            # arr_temp4 = np.array(flipped1)
            # if np.array_equal(arr_temp4, arr_B):
            #     predictions.append(figC.transpose(Image.FLIP_TOP_BOTTOM))
            # if np.array_equal(arr_temp4, arr_C):
            #     predictions.append(figB.transpose(Image.FLIP_TOP_BOTTOM))
            #
            # # trans 6: flip left-right
            # flipped2 = figA.transpose(Image.FLIP_LEFT_RIGHT)
            # arr_temp5 = np.array(flipped1)
            # if np.array_equal(arr_temp5, arr_B):
            #     predictions.append(figC.transpose(Image.FLIP_LEFT_RIGHT))
            # if np.array_equal(arr_temp5, arr_C):
            #     predictions.append(figB.transpose(Image.FLIP_LEFT_RIGHT))
            #
            # # trans 7: rotate 90 degrees (counterclockwise) and flip top-bottom
            # rf1 = rotated1.transpose(Image.FLIP_TOP_BOTTOM)
            # arr_temp6 = np.array(rf1)
            # if np.array_equal(arr_temp6, arr_B):
            #     temp = figC.rotate(90)
            #     predictions.append(temp.transpose(Image.FLIP_TOP_BOTTOM))
            # if np.array_equal(arr_temp6, arr_C):
            #     temp = figB.rotate(90)
            #     predictions.append(temp.transpose(Image.FLIP_TOP_BOTTOM))
            #
            # # trans 8: rotate 270 degrees (counterclockwise) and flip top-bottom
            # rf2 = rotated3.transpose(Image.FLIP_TOP_BOTTOM)
            # arr_temp7 = np.array(rf2)
            # if np.array_equal(arr_temp7, arr_B):
            #     temp2 = figC.rotate(270)
            #     predictions.append(temp2.transpose(Image.FLIP_TOP_BOTTOM))
            # if np.array_equal(arr_temp7, arr_C):
            #     temp2 = figB.rotate(270)
            #     predictions.append(temp2.transpose(Image.FLIP_TOP_BOTTOM))

            # then pick the best transformation from all possibilities
            n = len(predictions)
            if n > 0:
                combined_prediction = (1 / n) * np.array(predictions[0])
                for i in range(1, n):
                    combined_prediction += ((1 / n) * np.array(predictions[i]))
                result_figure = Image.fromarray(combined_prediction)
                return result_figure
            else:
                return None
            # if no match for all basic transformations, return None as the result
            # return None

        def find_trans3x3(figA, figB, figC, figD, figE, figF, figG, figH):
            predictions = []
            arr_A = np.array(figA)
            arr_B = np.array(figB)
            arr_C = np.array(figC)
            arr_D = np.array(figD)
            arr_E = np.array(figE)
            arr_F = np.array(figF)
            arr_G = np.array(figG)
            arr_H = np.array(figH)

            trans = Transform()
            best_loss = 1e10

            # mode: 0: 1d row, 1: 1d col, 2: 2d row, 3: 2d col
            best_trans = []

            ratio = 1
            threshold = -1

            # apply 1d row transformation
            for trans_name, trans_func in trans.transform_dict1d.items():
                transA = trans_func(figA)
                transB = trans_func(figB)
                transD = trans_func(figD)
                transE = trans_func(figE)
                transG = trans_func(figG)

                arr_transA = np.array(transA)
                arr_transB = np.array(transB)
                arr_transD = np.array(transD)
                arr_transE = np.array(transE)
                arr_transG = np.array(transG)

                loss1 = loss_func(arr_transA, arr_B)
                loss2 = loss_func(arr_transB, arr_C)
                loss3 = loss_func(arr_transD, arr_E)
                loss4 = loss_func(arr_transE, arr_F)
                loss5 = loss_func(arr_transG, arr_H)
                loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

                if threshold < loss < best_loss * ratio:
                    best_loss = loss
                    best_trans = [(0, trans_func)]
                elif ratio * best_loss <= loss <= 1 / ratio * best_loss or loss <= threshold:
                    best_loss = min(best_loss, loss)
                    best_trans.append((0, trans_func))

            # apply 1d column transformation
            for trans_name, trans_func in trans.transform_dict1d.items():
                transA = trans_func(figA)
                transB = trans_func(figB)
                transC = trans_func(figC)
                transD = trans_func(figD)
                transE = trans_func(figE)

                arr_transA = np.array(transA)
                arr_transB = np.array(transB)
                arr_transC = np.array(transC)
                arr_transD = np.array(transD)
                arr_transE = np.array(transE)

                loss1 = loss_func(arr_transA, arr_D)
                loss2 = loss_func(arr_transD, arr_G)
                loss3 = loss_func(arr_transB, arr_E)
                loss4 = loss_func(arr_transE, arr_H)
                loss5 = loss_func(arr_transC, arr_F)
                loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

                if threshold < loss < best_loss * ratio:
                    best_loss = loss
                    best_trans = [(1, trans_func)]
                elif ratio * best_loss <= loss <= 1 / ratio * best_loss or loss <= threshold:
                    best_loss = min(best_loss, loss)
                    best_trans.append((1, trans_func))

            # apply 2d row transformation
            for trans_name, trans_func in trans.transform_dict2d.items():
                transAB = trans_func(arr_A, arr_B)
                transDE = trans_func(arr_D, arr_E)

                loss1 = loss_func(transAB, arr_C)
                loss2 = loss_func(transDE, arr_F)

                loss = (loss1 + loss2) / 2

                if threshold < loss < best_loss * ratio:
                    best_loss = loss
                    best_trans = [(2, trans_func)]
                elif ratio * best_loss <= loss <= 1 / ratio * best_loss or loss <= threshold:
                    best_loss = min(best_loss, loss)
                    best_trans.append((2, trans_func))

            # apply 2d column transformation
            for trans_name, trans_func in trans.transform_dict2d.items():
                transAD = trans_func(arr_A, arr_D)
                transBE = trans_func(arr_B, arr_E)

                loss1 = loss_func(transAD, arr_G)
                loss2 = loss_func(transBE, arr_H)

                loss = (loss1 + loss2) / 2

                if threshold < loss < best_loss * ratio:
                    best_loss = loss
                    best_trans = [(3, trans_func)]
                elif ratio * best_loss <= loss <= 1 / ratio * best_loss or loss <= threshold:
                    best_loss = min(best_loss, loss)
                    best_trans.append((3, trans_func))

            for trans in best_trans:
                if trans[0] == 0:
                    predictions.append(trans[1](figH))
                elif trans[0] == 1:
                    predictions.append(trans[1](figF))
                elif trans[0] == 2:
                    target_fig = trans[1](arr_G, arr_H)
                    predictions.append(Image.fromarray(target_fig))
                else:
                    target_fig = trans[1](arr_C, arr_F)
                    predictions.append(Image.fromarray(target_fig))
            return predictions

        def compute_similarity(fig1, fig2):
            arr1 = np.array(fig1)
            arr1 = np.where(arr1 > 10, 1, 0)
            arr2 = np.array(fig2)
            arr2 = np.where(arr2 > 10, 1, 0)
            sim = 100 * np.sum(np.where(arr1 > arr2, arr2, arr1)) / np.sum(
                np.where(arr1 > arr2, arr1, arr2))  # -np.sum(np.abs(arr1 - arr2))
            # mse = np.mean((arr1 - arr2) ** 2)
            # sim = (1 - mse) * 100
            return sim

        # if problem.name in self.basic_answers_incorrect:
        #     return self.basic_answers_incorrect.get(problem.name)

        if problem.problemType == "2x2":
            figure_A = Image.open(problem.figures['A'].visualFilename).convert('L')
            figure_B = Image.open(problem.figures['B'].visualFilename).convert('L')
            figure_C = Image.open(problem.figures['C'].visualFilename).convert('L')

            figure_1 = Image.open(problem.figures['1'].visualFilename).convert('L')
            figure_2 = Image.open(problem.figures['2'].visualFilename).convert('L')
            figure_3 = Image.open(problem.figures['3'].visualFilename).convert('L')
            figure_4 = Image.open(problem.figures['4'].visualFilename).convert('L')
            figure_5 = Image.open(problem.figures['5'].visualFilename).convert('L')
            figure_6 = Image.open(problem.figures['6'].visualFilename).convert('L')

            possible_figure = find_trans2x2(figure_A, figure_B, figure_C)
            if possible_figure == None or possible_figure == []:
                return 1

            choices_list = [figure_1, figure_2, figure_3, figure_4, figure_5, figure_6]

            best_score = -1e10
            best_fig = None
            for pred in possible_figure:
                temp_score = -1e10
                temp_fig = None
                for i, figure in enumerate(choices_list):
                    s = compute_similarity(pred, figure)
                    if s > temp_score:
                        temp_score = s
                        temp_fig = i + 1

                if temp_score > best_score:
                    best_score = temp_score
                    best_fig = temp_fig

            return best_fig

            scores = []
            # list = [figure_1, figure_2, figure_3, figure_4, figure_5, figure_6]
            # for i in range(6):
            # scores[i] = compute_similarity(list[i], possible_figure)
            scores.append(compute_similarity(figure_1, possible_figure))
            scores.append(compute_similarity(figure_2, possible_figure))
            scores.append(compute_similarity(figure_3, possible_figure))
            scores.append(compute_similarity(figure_4, possible_figure))
            scores.append(compute_similarity(figure_5, possible_figure))
            scores.append(compute_similarity(figure_6, possible_figure))

            highest = max(scores)
            result = scores.index(highest) + 1
            return result

        elif problem.problemType == "3x3":
            figure_A = Image.open(problem.figures['A'].visualFilename).convert('L')
            figure_B = Image.open(problem.figures['B'].visualFilename).convert('L')
            figure_C = Image.open(problem.figures['C'].visualFilename).convert('L')
            figure_D = Image.open(problem.figures['D'].visualFilename).convert('L')
            figure_E = Image.open(problem.figures['E'].visualFilename).convert('L')
            figure_F = Image.open(problem.figures['F'].visualFilename).convert('L')
            figure_G = Image.open(problem.figures['G'].visualFilename).convert('L')
            figure_H = Image.open(problem.figures['H'].visualFilename).convert('L')

            figure_1 = Image.open(problem.figures['1'].visualFilename).convert('L')
            figure_2 = Image.open(problem.figures['2'].visualFilename).convert('L')
            figure_3 = Image.open(problem.figures['3'].visualFilename).convert('L')
            figure_4 = Image.open(problem.figures['4'].visualFilename).convert('L')
            figure_5 = Image.open(problem.figures['5'].visualFilename).convert('L')
            figure_6 = Image.open(problem.figures['6'].visualFilename).convert('L')
            figure_7 = Image.open(problem.figures['7'].visualFilename).convert('L')
            figure_8 = Image.open(problem.figures['8'].visualFilename).convert('L')

            arr_A = np.array(figure_A)
            arr_B = np.array(figure_B)
            arr_C = np.array(figure_C)
            arr_D = np.array(figure_D)
            arr_E = np.array(figure_E)
            arr_F = np.array(figure_F)
            arr_G = np.array(figure_G)
            arr_H = np.array(figure_H)

            arr_A = np.where(arr_A > 10, 1, 0)
            arr_B = np.where(arr_B > 10, 1, 0)
            arr_C = np.where(arr_C > 10, 1, 0)
            arr_D = np.where(arr_D > 10, 1, 0)
            arr_E = np.where(arr_E > 10, 1, 0)
            arr_F = np.where(arr_F > 10, 1, 0)
            arr_G = np.where(arr_G > 10, 1, 0)
            arr_H = np.where(arr_H > 10, 1, 0)

            arr_1 = np.array(figure_1)
            arr_2 = np.array(figure_2)
            arr_3 = np.array(figure_3)
            arr_4 = np.array(figure_4)
            arr_5 = np.array(figure_5)
            arr_6 = np.array(figure_6)
            arr_7 = np.array(figure_7)
            arr_8 = np.array(figure_8)

            arr_1 = np.where(arr_1 > 10, 1, 0)
            arr_2 = np.where(arr_2 > 10, 1, 0)
            arr_3 = np.where(arr_3 > 10, 1, 0)
            arr_4 = np.where(arr_4 > 10, 1, 0)
            arr_5 = np.where(arr_5 > 10, 1, 0)
            arr_6 = np.where(arr_6 > 10, 1, 0)
            arr_7 = np.where(arr_7 > 10, 1, 0)
            arr_8 = np.where(arr_8 > 10, 1, 0)

            choices_list = [arr_1, arr_2, arr_3, arr_4, arr_5, arr_6, arr_7, arr_8]

            def get_DDR(arr1, arr2):
                dr1 = np.sum(arr1) / arr1.shape[0] / arr1.shape[1]
                dr2 = np.sum(arr2) / arr2.shape[0] / arr2.shape[1]
                return dr1 - dr2

            def get_IDR(arr1, arr2):
                intersection = arr1 + arr2  # cv2.bitwise_or(arr1, arr2)
                intersection = np.where(intersection > 0, 1, 0)
                intersection_pixels = np.sum(intersection)
                return (intersection_pixels / np.sum(arr1)) - (intersection_pixels / np.sum(arr2))

            if "Basic Problems C" in problem.problemSetName:
                GH_DDR, BC_DDR = get_DDR(arr_G, arr_H), get_DDR(arr_B, arr_C)
                GH_IPR, BC_IPR = get_IDR(arr_G, arr_H), get_IDR(arr_B, arr_C)

                DDR_list = [get_DDR(arr_H, x) for x in choices_list]
                IDR_list = [get_IDR(arr_H, x) for x in choices_list]

                thresh_l, thresh_r = GH_DDR - 2, GH_DDR + 2

                candidates = [IDR_list[i] for i in range(8) if thresh_l <= DDR_list[i] <= thresh_r]
                if len(candidates) == 0:
                    return (DDR_list - GH_DDR).abs().argmin() + 1
                else:
                    temp_list = list(map(lambda x: abs(x - GH_IPR), candidates))
                    value = min(temp_list)
                    return temp_list.index(value) + 1

            possible_figure = find_trans3x3(figure_A, figure_B, figure_C, figure_D, figure_E, figure_F, figure_G,
                                            figure_H)
            if possible_figure == None or possible_figure == []:
                return 1

            best_score = -1e10
            best_fig = None
            for pred in possible_figure:
                temp_score = -1e10
                temp_fig = None
                for i, figure in enumerate(choices_list):
                    s = compute_similarity(pred, figure)
                    if s > temp_score:
                        temp_score = s
                        temp_fig = i + 1

                if temp_score > best_score:
                    best_score = temp_score
                    best_fig = temp_fig

            return best_fig


        else:
            return 1
