"""
FormTemplate.py
Written by Adam Lavertu
Stanford University
"""

import os
import numpy as np
from glob import glob

import json

from skimage import io, img_as_float
from skimage.feature import match_template
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

import imreg_dft as ird

from pystackreg import StackReg

from ImageDataTools import *
from ProcessingUtils import prep_image_data

import torch


class FormTemplate(object):
    def __init__(self, path_to_form_dir, prototyping=False):

        path_to_template = glob(os.path.join(path_to_form_dir, "*_cmr.png"))[0]


        self.isCal = "cal" in path_to_template
        self.prototyping = prototyping

        self.template_image = io.imread(path_to_template).astype(float)

        if np.max(self.template_image) > 1.0:
            self.template_image = img_as_float(self.template_image)

        if np.max(self.template_image) > 1.0:
            self.template_image = self.template_image/255.0

        self.gray_template = self.preprocess_image(self.template_image)

        self.binary_template = np.zeros(self.gray_template.shape)

        self.binary_template[
            self.gray_template > threshold_otsu(self.gray_template)
        ] = 1.0



        self.high_pr_boxes = json.load(
            open(glob(os.path.join(path_to_form_dir, "*_high_pr_prop.json"))[0], "r")
        )
        self.cong_boxes = json.load(
            open(
                glob(os.path.join(path_to_form_dir, "*_cong_setting_coords.json"))[0],
                "r",
            )
        )
        one_box = list(self.high_pr_boxes.values())[0]
        self.chk_buffer =  (one_box[1] - one_box[0])//2

        self.num_high_pr = len(self.high_pr_boxes)
        self.num_cong_boxes = len(self.cong_boxes)

        self.template_checkboxes = self.get_form_checkboxes(self.gray_template)


        self.sr1 = StackReg(StackReg.SCALED_ROTATION)
        self.sr2 = StackReg(StackReg.AFFINE)

        section_h = self.gray_template.shape[0]//5
        section_w = self.gray_template.shape[1]//5
        self.alignment_template = self.gray_template[(section_w*1):(section_w*4), (section_h*1):(section_h*4)]
        self.ref_x, self.ref_y = self.get_template_coords(self.gray_template, self.alignment_template)



    def get_template_coords(self, reg_im, temp_template):
        temp_matches = match_template(reg_im, temp_template)
        ij = np.unravel_index(np.argmax(temp_matches), temp_matches.shape)
        x, y = ij[::-1]
        return(x,y)

    def alignment_dist_metric(self, reg_im):
        x, y = self.get_template_coords(reg_im, self.alignment_template)
        score = np.sum([abs(x-self.ref_x), abs(y-self.ref_y)])
        return(score)

    def preprocess_image(self, image_in):
        image1 = rgb2gray(image_in)

        if np.max(image1) > 5.0:
            image1 = img_as_float(image1)/255.0

        return(image1)

    def rescale_2_template_size(self, im_to_resize):
        image1 = resize(im_to_resize, self.gray_template.shape)
        return image1

    def check_template_match(self, reg_can_image):
        match_coord_distance = self.alignment_dist_metric(reg_can_image)
        match_peak_score = np.max(match_template(reg_can_image, self.alignment_template))

        if self.prototyping:
            print("match score:", [match_coord_distance, match_peak_score])
        return([match_coord_distance, match_peak_score])

    def rotate_scale_then_affine_registration(self, template, image1):

        binary_template = np.zeros(template.shape)
        binary_template[template > threshold_otsu(template)] = 1.0

        image1_binary = np.zeros(image1.shape)
        image1_binary[image1 > threshold_otsu(image1)] = 1.0

        tmat1 = self.sr1.register(binary_template, image1_binary)
        temp_bin_transform_1 = self.sr1.transform(image1_binary, tmat=tmat1)
        tmat2 = self.sr2.register(binary_template, temp_bin_transform_1)

        reg_im = self.sr1.transform(image1, tmat=tmat1)
        reg_im = self.sr2.transform(reg_im, tmat=tmat2)

        return(reg_im)

    def get_safe_coord_shift(self, image, desired_buffer, xmi, xma, ymi, yma):
        xmi_ms = xmi
        xma_ms = image.shape[0] - xma
        x_ms = np.min([xmi_ms, xma_ms])

        if desired_buffer > x_ms:
            xmi = xmi - x_ms
            xma = xma + x_ms
        else:
            xmi = xmi - desired_buffer
            xma = xma + desired_buffer

        ymi_ms = ymi
        yma_ms = image.shape[0] - yma
        y_ms = np.min([ymi_ms, yma_ms])

        if desired_buffer > y_ms:
            ymi = ymi - y_ms
            yma = yma + y_ms
        else:
            ymi = ymi - desired_buffer
            yma = yma + desired_buffer

        return [xmi, xma, ymi, yma]

    def register_form_pystack(self, image1):

        image1_binary = np.zeros(image1.shape)
        image1_binary[image1 > threshold_otsu(image1)] = 1.0

        tmat1 = self.sr1.register(self.binary_template, image1_binary)
        temp_bin_transform_1 = self.sr1.transform(image1_binary, tmat=tmat1)
        tmat2 = self.sr2.register(self.binary_template, temp_bin_transform_1)

        reg_im1 = self.sr1.transform(image1, tmat=tmat1)
        reg_im2 = self.sr2.transform(reg_im1, tmat=tmat2)

        return([reg_im1, reg_im2])

    def register_form_ird(self, image1, num_iter=50, func_order = 3, threshold=0.5):

        try:
            prev_score = 0.0
            for j in range(num_iter):
                result = ird.similarity(self.gray_template, image1, numiter=1, order=func_order)
                image1 = result['timg']
                loc_distance, sim_score = self.check_template_match(image1)

                if sim_score > threshold or sim_score < prev_score:
                    break

                prev_score = sim_score
        except ValueError:
            pass

        reg_im = image1
        return(reg_im)

    def register_2_template(self, image_in, threshold=0.50, distance_threshold=200):

        im_to_align = self.preprocess_image(image_in)

        if im_to_align.shape != self.gray_template.shape:
            im_to_align = self.rescale_2_template_size(im_to_align)

        pystack_reg1, pystack_reg2 = self.register_form_pystack(im_to_align)

        pystack_reg1_distance, pystack_reg1_score = self.check_template_match(pystack_reg1)
        pystack_reg2_distance, pystack_reg2_score = self.check_template_match(pystack_reg2)

        if pystack_reg1_score < threshold and pystack_reg2_score < threshold:

            ird_reg = self.register_form_ird(im_to_align, threshold=threshold/2)
            ird_reg_distance, ird_reg_score = self.check_template_match(ird_reg)

            if ird_reg_score > threshold/2 and ird_reg_distance < distance_threshold:
                return(ird_reg)
            else:
                return(None)

        if pystack_reg2_score >= pystack_reg1_score:
            return(pystack_reg2)
        else:
            return(pystack_reg1)

    def get_form_checkboxes(self, reg_im, context_buffer_percentage = 0.15, shift_right=20):
        out_checkboxes = []

        reg_im = self.preprocess_image(reg_im)

        context_buffer = int(np.min(reg_im.shape) * context_buffer_percentage)
        context_buffer2 = context_buffer//8
        context_buffer3 = self.chk_buffer

        for xmi, xma, ymi, yma in self.high_pr_boxes.values():
            buff_xmi, buff_xma, buff_ymi, buff_yma = self.get_safe_coord_shift(
                reg_im, context_buffer, xmi, xma, ymi, yma
            )
            bigger_self = reg_im[buff_xmi:buff_xma, buff_ymi:buff_yma]

            buff2_xmi, buff2_xma, _, _ = self.get_safe_coord_shift(
                reg_im, context_buffer3, xmi, xma, ymi, yma
            )
            _, _, buff2_ymi, buff2_yma = self.get_safe_coord_shift(
                reg_im, context_buffer2, xmi, xma,(ymi+shift_right), (yma+shift_right)
            )
            smaller_template = reg_im[buff2_xmi:buff2_xma, buff2_ymi:buff2_yma]

            smaller_template2 = reg_im[xmi:xma, (ymi+shift_right):(yma+shift_right)]

            out_checkboxes.append([xmi, xma, ymi, yma, bigger_self,smaller_template, smaller_template2])

        for xmi, xma, ymi, yma in self.cong_boxes.values():
            buff_xmi, buff_xma, buff_ymi, buff_yma = self.get_safe_coord_shift(
                reg_im, context_buffer, xmi, xma, ymi, yma
            )
            bigger_self = reg_im[buff_xmi:buff_xma, buff_ymi:buff_yma]

            buff2_xmi, buff2_xma, _, _ = self.get_safe_coord_shift(
                reg_im, context_buffer3, xmi, xma, ymi, yma
            )
            _, _, buff2_ymi, buff2_yma = self.get_safe_coord_shift(
                reg_im, context_buffer2, xmi, xma, (ymi+shift_right), (yma+shift_right)
            )

            smaller_template = reg_im[buff2_xmi:buff2_xma, buff2_ymi:buff2_yma]

            smaller_template2 = reg_im[xmi:xma, (ymi+shift_right):(yma+shift_right)]

            out_checkboxes.append([xmi, xma, ymi, yma, bigger_self,smaller_template, smaller_template2])

        return out_checkboxes

    def search_for_form_checkboxes(self, reg_im, search_buffer_percentage=0.2, shift_right=20):

        reg_im = self.preprocess_image(reg_im)

        search_buffer = int(np.min(reg_im.shape) * search_buffer_percentage)
        if self.prototyping:
            print("Search_buffer:", search_buffer)


        out_checkboxes = []
        for j,(xmi, xma, ymi, yma, chk_template, chk_template2, chk_only_template) in enumerate(self.template_checkboxes):

            xmi += shift_right
            xma += shift_right

            buff_xmi, buff_xma, buff_ymi, buff_yma = self.get_safe_coord_shift(
                reg_im, search_buffer, xmi, xma, ymi, yma
            )


            if self.prototyping:
                mi_buff_size = np.min([buff_xma-buff_xmi, buff_yma-buff_ymi])
                print(mi_buff_size)

            patch_2_search = reg_im[buff_xmi:buff_xma, buff_ymi:buff_yma]
            checkbox_matches = match_template(patch_2_search, chk_template, pad_input=True)

            ij = np.unravel_index(np.argmax(checkbox_matches), checkbox_matches.shape)
            x, y = ij[::-1]

            # Initiate search round 2
            h, w = search_buffer//3, search_buffer//3#self.chk_buffer, self.chk_buffer

            xmi_out = np.max([0, (y-w)])
            xma_out = np.min([patch_2_search.shape[0], (y+w)])
            ymi_out = np.max([0, (x-h)])
            yma_out = np.min([patch_2_search.shape[1], (x+h)])

            smaller_patch_2_search = patch_2_search[xmi_out:xma_out,ymi_out:yma_out]

            secondary_checkbox_matches = match_template(smaller_patch_2_search, chk_template2, pad_input=True)

            ij = np.unravel_index(np.argmax(secondary_checkbox_matches), secondary_checkbox_matches.shape)
            x, y = ij[::-1]

            # Now we're zerod in on the right region

            # Initiate search round 2
            h, w = self.chk_buffer, self.chk_buffer
            xmi_out = np.max([0, (y-w)])
            xma_out = np.min([smaller_patch_2_search.shape[0], (y+w)])
            ymi_out = np.max([0, (x-h)])
            yma_out = np.min([smaller_patch_2_search.shape[1], (x+h)])

            relevant_box = smaller_patch_2_search[xmi_out:xma_out,(ymi_out-shift_right):(yma_out-shift_right)]

            # This is a temporary patch, just to ignore poorly extracted checkboxes
            # if relevant_box.shape[0] == 0 or relevant_box.shape[1] == 0:
            #     relevant_box = np.zeros((self.chk_buffer, self.chk_buffer))
            out_checkboxes.append(relevant_box)

        return out_checkboxes


    def process_chkboxes(self, device, transform, ens_model, reg_im, opt_thres=0.5):
        hp = False
        cs = False
        unc = False

        check_data = self.search_for_form_checkboxes(reg_im)
        # print('max, min pixel value after extraction:', np.max(check_data[0]), np.min(check_data[0]))

        imgs = prep_image_data(check_data, transform)

        # if self.prototyping:
        #     proc_imgs = [x[0,:,:] for x in imgs]
        #     grid_images(proc_imgs, num_cols=2)

        imgs = imgs.to(device, dtype=torch.float)
        preds = ens_model(imgs)
        preds = torch.sigmoid(preds)

        if self.prototyping:
            print(np.around(preds.detach().numpy(), decimals=2))

        hp = any(
            [
                True if x else False
                for j, x in enumerate(
                    ((preds[:self.num_high_pr, :] >= opt_thres).sum(dim=1) >= 2).tolist()
                )
            ]
        )
        cs = any(
            [
                True if x else False
                for j, x in enumerate(
                    ((preds[self.num_high_pr:, :] >= opt_thres).sum(dim=1) >= 2).tolist()
                )
            ]
        )

        if self.prototyping:
            print(hp, cs)
        return [hp, cs, unc]
