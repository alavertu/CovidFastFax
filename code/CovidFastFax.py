"""
CovidFastFax.py
Written by Adam Lavertu
Stanford University
"""

import argparse
import requests
import time
import shutil
import warnings
import sys

warnings.simplefilter(action="ignore", category=FutureWarning)

from skimage import img_as_float32
from pdf2image import convert_from_path

from FormTemplate import *

from models import *

from ProcessingUtils import check_directory, load_model


class CovidFastFax(object):
    def __init__(
        self,
        target_dir,
        output_dir,
        model_module=None,
        reset=False,
        forced_reset=False,
        verbose=False,
        debug_mode=False,
        split_pdfs=False,
        email_alerts=False,
    ):

        self.target_dir = target_dir

        self.output_dir = output_dir
        self.verbose = verbose
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.debug_mode_dir = "./debugging_data"
            check_directory(self.debug_mode_dir)
            self.outdebug_mode = open(
                os.path.join(self.debug_mode_dir, "debug_mode_output.csv"), "w+"
            )
            self.outdebug_mode.write(f"file,pred,registration_score,page\n")

        self.split_pdfs = split_pdfs

        self.email_out = email_alerts
        self.email_ping_rate = 20
        self.email_time_tracker = 0.0
        if self.verbose:
            print(f"email setting: {self.email_out}")

        self.email_server = json.load(open("./email_endpoint.json", "r"))["email_url"]
        if self.email_out:
            if self.verbose:
                print(f"Email pings will be sent to: {self.email_server}")
            if self.email_server == "INSERT_EMAIL_SERVER_URL_HERE":
                print(f"ERROR: Have you setup the url endpoint for email pings?")
                sys.exit()

        if self.verbose:
            print("Starting program...")

        if reset:
            self.processed = set()
            if self.verbose:
                print("Resetting cache...")
            self.cache1 = open("./cache1.txt", "w+")
            self.cache2 = open("./cache2.txt", "w+")
            if forced_reset:
                if os.path.exists(self.output_dir):
                    shutil.rmtree(self.output_dir)
        else:
            self.processed = set()
            if self.verbose:
                print("Reading cache...")
            if os.path.exists("./cache1.txt"):
                with open("./cache1.txt") as inFile:
                    for line in inFile:
                        self.processed.add(line.strip())
            if os.path.exists("./cache2.txt"):
                with open("./cache2.txt") as inFile:
                    for line in inFile:
                        self.processed.add(line.strip())
            if self.verbose:
                print(f"Found {len(self.processed)} files in the cache...")
            self.cache1 = open("./cache1.txt", "a+")
            self.cache2 = open("./cache2.txt", "a+")

        self.skipped = open("./skipped.txt", "a+")

        self.hcw_case_dir = "./healthcare_workers"
        self.cong_setting_dir = "./congregate_settings"

        for pth in [
            self.output_dir,
            self.hcw_case_dir,
            self.cong_setting_dir,
        ]:
            check_directory(pth)

        # Device configuration
        _ = torch.no_grad()
        self.device = torch.device("cpu")

        ## SETUP FOR FORM CLASSIFICATION
        self.template_classifiers = []
        for model_path in glob(os.path.join("../data/form_model/base_models/", "*.pt")):
            self.template_classifiers.append(self.load_template_model(model_path))

        if model_module is not None:
            for model_path in glob(os.path.join(model_module, "*.pt")):
                self.template_classifiers.append(self.load_template_model(model_path))

        form_mu = (0.9094463458110517,)
        form_std = (0.1274794325726292,)
        self.form_transf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((550, 425)),
                transforms.ToTensor(),
                transforms.Normalize(form_mu, form_std),
            ]
        )

        ### SETUP FOR CHECKBOX COMPONENT
        check_mu = (0.010199239219432315,)
        check_std = (0.23450328975030799,)
        self.chk_transf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((50, 50)),
                transforms.ToTensor(),
                transforms.Normalize(check_mu, check_std),
            ]
        )

        # Need to update checkbox coordinates

        # Load models into ensemble predictor
        model1 = load_model("../data/chkbox_models/chkbox_rnd11_0_bestVal.pt")
        model2 = load_model("../data/chkbox_models/chkbox_rnd11_1_bestVal.pt")
        model3 = load_model("../data/chkbox_models/chkbox_rnd11_2_bestVal.pt")
        model4 = load_model("../data/chkbox_models/chkbox_rnd11_3_bestVal.pt")
        model5 = load_model("../data/chkbox_models/chkbox_rnd11_4_bestVal.pt")

        _ = model1.to(self.device)
        _ = model2.to(self.device)
        _ = model3.to(self.device)
        _ = model4.to(self.device)
        _ = model5.to(self.device)

        self.ens_model = CheckBoxEnsemble(model1, model2, model3, model4, model5)
        _ = self.ens_model.to(self.device)
        _ = self.ens_model.eval()

        # SETUP FOR FORM TEMPLATES
        self.templates = dict()
        for template in sorted(glob("../data/references/*")):
            self.templates[os.path.basename(template)] = FormTemplate(
                template, prototyping=self.debug_mode
            )

    def load_template_model(self, path_2_model):
        model_pred_key = {int(k):v for k,v in json.load(open(os.path.join(os.path.dirname(path_2_model), "model_pred_key.json"))).items()}
        template_model = TemplateNet(num_classes=len(model_pred_key))
        template_model.load_state_dict(
            torch.load(
                path_2_model, map_location=torch.device("cpu")
            )
        )
        _ = template_model.to(self.device)
        _ = template_model.eval()
        return([template_model, model_pred_key])

    def email_ping(self, time_elapsed):

        if self.email_out:
            self.email_time_tracker += time_elapsed

            if self.email_time_tracker >= self.email_ping_rate:
                if self.verbose:
                    print(f"Pinging email server: {self.email_server}")
                _ = requests.get(self.email_server)
                self.email_time_tracker = 0.0

    def monitor(self):
        if self.verbose:
            print(f"Monitoring {self.target_dir} ...")
        while True:
            targ_files = set(glob(os.path.join(self.target_dir, "*.pdf"))).union(
                glob(os.path.join(self.target_dir, "*.PDF"))
            )
            to_process = targ_files.difference(self.processed)
            self.process_file_list(to_process)
            time.sleep(60)
            self.email_ping(1)
            if self.verbose:
                print(f"Monitoring {self.target_dir} for new files...")

    def process_file_list(self, to_process):
        for file_path in to_process:
            if self.verbose:
                print(f"Analyzing {os.path.basename(file_path)}...")
            start = time.time()
            self.process_pdf(file_path)
            end = time.time()
            
            time_elapsed = ((end - start)/60)
            if self.verbose:
                print(f"Processing time: {round(time_elapsed*60, 2)} seconds")

            self.email_ping(time_elapsed)
            self.processed.add(file_path)
            self.cache1.write(file_path + "\n")
            self.cache2.write(file_path + "\n")

    def register_form_pystack(self, image_in, form_template):
        reg_im = form_template.register_2_template(np.asarray(image_in))
        return reg_im

    def get_form_registration(self, image, temp_template):
        reg_im = temp_template.register_2_template(np.asarray(image))

        return reg_im

    def save_to_pdf(self, base_image, out_path, other_pages=[]):
        if len(other_pages) > 0:
            _ = base_image.save(
                out_path,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=other_pages,
            )
        else:
            _ = base_image.save(out_path, "PDF", resolution=100.0, save_all=True)

    def get_report_priority(self, temp_im, form_template):
        if len(form_template.template_checkboxes) > 0:
            checkbox_preds = form_template.process_chkboxes(
                self.device, self.chk_transf, self.ens_model, temp_im
            )
            return np.array(checkbox_preds)
        else:
            return np.array(list())

    def create_image_stacks(self, file_path, f_baseroot):
        og_im_stack = []
        proc_im_stack = []

        try:
            if os.path.exists(file_path):
                images = convert_from_path(file_path)
            else:
                self.skipped.write(file_path + "\n")
                return [None, None]

        except ValueError:
            if os.path.exists(file_path):
                temp_name = f_baseroot + f"_NotProcessed.pdf"
                shutil.copy(file_path, os.path.join(self.output_dir, temp_name))
                if self.verbose:
                    print(f"{file_path} was not processed due to an error.")
            return [None, None]

        for j, im in enumerate(images):
            # Store the original
            og_im_stack.append(im)
            temp_im = rgb2gray(np.asarray(im))
            temp_im = img_as_float32(temp_im)
            im_in = self.form_transf(temp_im)
            proc_im_stack.append(im_in)

        return [og_im_stack, proc_im_stack]

    def get_form_template_classifications(self, proc_im_stack):

        # Pass pages through template matcher to identify forms vs. labs etc...
        proc_im_stack = torch.stack(proc_im_stack)
        proc_im_stack = proc_im_stack.to(self.device)

        pred_labels = [set() for _ in range(proc_im_stack.shape[0])]
        for k, (model, pred_key) in enumerate(self.template_classifiers):
            form_type_preds = model(proc_im_stack)
            sf_preds = torch.nn.functional.softmax(form_type_preds, dim=1)
            for j in range(sf_preds.shape[0]):
                temp_labels = {
                    pred_key.get(x)
                    for x in np.where(sf_preds[j, :] > 0.5)[0]
                    if x != 0
                }
                pred_labels[j].update(temp_labels)
                if self.debug_mode:
                    print(f"Model {k}: ", [round(x.item(), 2) for x in sf_preds[j, :]])

        return pred_labels

    def verify_pred_label(self, pred_labels, og_im_stack, page_num):
        form_match = False
        for pred in pred_labels:
            if self.debug_mode:
                print(f"Pred label:{page_num}, {pred}")

            temp_template = self.templates.get(pred)
            if temp_template is None:
                print(self.templates)
                print(pred)
                print("FATAL ERROR: Template not in known templates...")
                exit()

            reg_im = self.get_form_registration(og_im_stack[page_num], temp_template)

            if reg_im is not None:
                # _ = plt.imshow(1.0-reg_im, cmap='Greys', vmin=0, vmax=1)
                # _ = plt.show()
                break

            elif pred == "ccc_april_2020":
                temp_template = self.templates.get("ccc_march_2020")
                reg_im = self.get_form_registration(
                    og_im_stack[page_num], temp_template
                )
                # grid_images([temp_template.gray_template, reg_im])
                if reg_im is not None:
                    pred = "ccc_march_2020"
                    break

            elif pred == "ccc_march_2020":
                temp_template = self.templates.get("ccc_april_2020")
                reg_im = self.get_form_registration(
                    og_im_stack[page_num], temp_template
                )
                if reg_im is not None:
                    pred = "ccc_april_2020"
                    break

        return [pred, reg_im, temp_template]

    def get_form_data(self, pred_labels, og_im_stack):

        # Create cover image metadata placeholder, defaults to type
        start_page = 0

        # Check if this is for jmi reports, if so, save reports in memory before outputting them,
        # So that the reports can be rearranged approriately
        hit_form_tracker = []

        for k, x in enumerate(pred_labels):
            if len(x) != 0:
                (
                    pred,
                    reg_im,
                    temp_template,
                ) = self.verify_pred_label(x, og_im_stack, k)

                if self.debug_mode:
                    if reg_im is not None:
                        reg_distance, reg_score = temp_template.check_template_match(
                            reg_im
                        )
                        # print(reg_score)
                        self.outdebug_mode.write(f"{pred},{reg_score},{k}\n")
                    # _ = skimage.io.imsave(os.path.join(self.debug_mode_dir, f'{f_baseroot}_regIm.png'), reg_im)

                if reg_im is not None:
                    if self.debug_mode:
                        print(f"Match  label:{k}, {pred}")

                    start_page = k

                    checkbox_preds = self.get_report_priority(reg_im, temp_template)

                    hit_form_tracker.append([start_page, pred, checkbox_preds])

        return hit_form_tracker

    def generate_output(self, file_path, f_baseroot, og_image_stack, hit_form_info):

        if self.debug_mode:
            print(hit_form_info)

        jmi_mode = "jm_v1" in {x[1] for x in hit_form_info}

        if len(hit_form_info) == 0:
            if os.path.exists(file_path):
                temp_name = f"NTD_{f_baseroot}.pdf"
                shutil.copy(file_path, os.path.join(self.output_dir, temp_name))
                if self.verbose:
                    print(f"No templates detected in {file_path}...")
        else:
            # Specific handling of jmi reports which has a three page format, assumes all samples are jmi style reports
            # I hate this, as its hardcoded, will rework solution with the output mode as part of the FormTemplate class
            if (
                jmi_mode
                and self.split_pdfs
                and (
                    len(og_image_stack) == len(hit_form_info) * 3
                    or (len(og_image_stack) - 1) == len(hit_form_info) * 3
                )
            ):
                if self.verbose:
                    print(
                        f"Splitting {file_path} into {len(hit_form_info)} separate report files"
                    )

                lead_page = (len(og_image_stack) - 1) == len(hit_form_info) * 3

                for index, (
                    page_num,
                    form_type,
                    checkbox_status
                ) in enumerate(hit_form_info):

                    other_pages = []

                    # Add report page and the additional other pages in proper order
                    other_pages.append(og_image_stack[page_num])
                    other_pages.append(og_image_stack[(page_num + 1)])

                    # Check for fax cover letter, if there is one then output that cover letter with the first sample
                    if lead_page and index == 0 and page_num == 2:
                        other_pages.append(og_image_stack[0])

                    # Check for a single trailing page
                    elif (
                        lead_page
                        and index == (len(hit_form_info) - 1)
                        and page_num == (len(hit_form_info) - 1)
                    ):
                        other_pages.append(og_image_stack[-1])

                    if any(checkbox_status >= 1):
                        check_hit_index = np.argmax(checkbox_status >= 1)
                        temp_name = f"{self.templates[form_type].checkbox_labels[check_hit_index]}_{f_baseroot}_{index+1}_of_{len(hit_form_info)}.pdf"
                    else:
                        temp_name = (
                            f"{self.templates[form_type].baseline_name}_{f_baseroot}_{index+1}_of_{len(hit_form_info)}.pdf"
                        )
                    regular_out = os.path.join(self.output_dir, temp_name)
                    self.save_to_pdf(
                        og_image_stack[(page_num - 1)], regular_out, other_pages
                    )

            elif self.split_pdfs and (
                len(og_image_stack) == len(hit_form_info)
                or (len(og_image_stack) - 1) == len(hit_form_info)
            ):
                if self.verbose:
                    print(
                        f"Splitting {file_path} into {len(hit_form_info)} separate report files"
                    )

                lead_page = (len(og_image_stack) - 1) == len(hit_form_info)

                for index, (
                    page_num,
                    form_type,
                    checkbox_status
                ) in enumerate(hit_form_info):
                    if self.debug_mode:
                        print(page_num, form_type, checkbox_status)

                    other_pages = []

                    # Check for fax cover letter, if there is one then output that cover letter with the first sample
                    if lead_page and index == 0 and page_num == 1:
                        other_pages.append(og_image_stack[0])

                    # Check for a single trailing page
                    elif (
                        lead_page
                        and index == (len(hit_form_info) - 1)
                        and page_num == (len(hit_form_info) - 1)
                    ):
                        other_pages.append(og_image_stack[-1])

                    if any(checkbox_status >= 1):
                        check_hit_index = np.argmax(checkbox_status >= 1)
                        temp_name = f"{self.templates[form_type].checkbox_labels[check_hit_index]}_{f_baseroot}_{index+1}_of_{len(hit_form_info)}.pdf"

                    else:
                        temp_name = (
                            f"{self.templates[form_type].baseline_name}_{f_baseroot}_{index+1}_of_{len(hit_form_info)}.pdf"
                        )

                    regular_out = os.path.join(self.output_dir, temp_name)
                    self.save_to_pdf(og_image_stack[page_num], regular_out, other_pages)

            else:
                if self.verbose:
                    print(f"{file_path} contained {len(hit_form_info)} reports...")
                # Assign page specific labels

                report_pages = []
                prefixes = []
                for page, form_type, checkbox_status in hit_form_info:
                    if any(checkbox_status >= 1):
                        check_hit_index = np.argmax(checkbox_status >= 1)

                        # Use first letter of output name as page prefix
                        prefix = self.templates[form_type].checkbox_labels[check_hit_index]
                        prefixes.append(self.templates[form_type].checkbox_labels[check_hit_index])

                        p_label = prefix.split("_")[1][0]
                        report_pages.append(p_label + str(page + 1))
                    else:
                        prefixes.append(self.templates[form_type].baseline_name)
                        report_pages.append(str(page + 1))

                report_pages = "-".join(report_pages)

                highest_pr_prefix = sorted(prefixes)[0]
                temp_name = f"{highest_pr_prefix}_{f_baseroot}_{len(hit_form_info)}_samples_pgs_{report_pages}.pdf"

                shutil.copy(file_path, os.path.join(self.output_dir, temp_name))

    def process_pdf(self, file_path):
        f_baseroot = os.path.basename(file_path).split(".")[0]
        og_im_stack, proc_im_stack = self.create_image_stacks(file_path, f_baseroot)
        if og_im_stack is not None:
            pred_labels = self.get_form_template_classifications(proc_im_stack)
            hit_form_info = self.get_form_data(pred_labels, og_im_stack)
            self.generate_output(file_path, f_baseroot, og_im_stack, hit_form_info)


"""
Parse the command line
"""


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Covid Fast Fax with basic checkbox prioritization"
    )
    requiredNamed = parser.add_argument_group("required arguments")
    requiredNamed.add_argument(
        "-t",
        "--target_dir",
        help="Target directory to monitor for new PDFs",
        required=True,
    )
    requiredNamed.add_argument(
        "-O", "--output_dir", help="Location to create output directory", required=True
    )
    parser.add_argument(
        "-m",
        "--model_module",
        help="Path to directory containing additional template detection models (for expanding to new templates)"
    )
    parser.add_argument(
        "-r",
        "--reset",
        help="Reset cache, will result in reprocessing of all files in the target directory",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--forced_reset",
        help="Reformat target directory, use with extreme caution, -r flag must also be specified",
        action="store_true",
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbose mode", action="store_true", default=False
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Debugging mode",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--split_pdfs",
        help="Split PDFs if a CMR template is detected. Creates a new PDF for each detected PDF, if the evaluated PDF "
        + " only contains CMR pages, plus or minus one page.",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--email_pings",
        help="Send still alive alerts by pinging the server in email_endpoint.json",
        action="store_true",
    )
    options = parser.parse_args()
    return options


"""
Main
"""
if __name__ == "__main__":
    options = parse_command_line()
    CRR = CovidFastFax(
        options.target_dir,
        options.output_dir,
        options.model_module,
        options.reset,
        options.forced_reset,
        options.verbose,
        options.debug,
        options.split_pdfs,
        options.email_pings,
    )
    CRR.monitor()
