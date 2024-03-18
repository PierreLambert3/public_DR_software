from engine.gui.container import Container
from engine.gui.scatterplot import Scatterplot, Heatmap
from engine.gui.window import Window
from engine.gui.selector import  Button, Mutex_with_title, String_selector, Number_selector, Mutex_choice, Scrollable_bundle
from engine.gui.listener import Listener
from engine.gui.graph import QA_graph
from engine.gui.event_ids import *
from utils import random_colors
from utils import luminosity_change
import numpy as np

class Main_screen():
    def __init__(self, theme, window, manager):
        self.manager = manager
        self.color = theme["color"]; self.background_color = theme["background"]
        self.eight_colors = random_colors(8)
        self.main_view   = Container((0,0), (0.745, 0.75), "scatterplots", window, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.right_view  = Container((0.75,0), (0.25, 1), "hparams", window, manager.uid_generator, color=theme["color"], background_color=theme["background"])
        self.bottom_view = Container((0,0.75), (0.75, 0.25), "bottom QA", window, manager.uid_generator, color=theme["color"], background_color=theme["background"])
        self.right_view.add_line( [(0,0.1), (0,0.675), 1, self.right_view.color])
        window.add_container(self.main_view)
        window.add_container(self.right_view)
        window.add_container(self.bottom_view)

        self.dataset_tabs = Mutex_choice((0,0), (1,0.1), "dataset tabs", self.main_view, uid_generator=self.manager.uid_generator, nb_col=-1, labels = ["new"], on_value_change_listener=Listener(DATASET_SELECTED, [self.manager]), ignore_with_ctrl=True, color = np.array([0, 150, 255]))
        self.main_view.add_leaf(self.dataset_tabs)
        self.dataset_tabs.notify_on_Rclick   = True
        self.dataset_tabs.on_Rclick_listener = Listener(DELETED_DATASET, [self.manager])
        self.selected_tab = self.dataset_tabs.value
        # scatterplot_fields: dictionnary with dataset/subset name as key and container as value (exept for "new: directly a Mutex_choice")
        new_dataset_field = Container((0,0.1), (1,0.9), "scatterplot field", self.main_view, self.manager.uid_generator)
        new_dataset_field.add_leaf(Mutex_choice((0,0),(1,1),"dataset mutex", new_dataset_field, uid_generator=self.manager.uid_generator, nb_col=3, labels = self.manager.dataset_names, on_value_change_listener=Listener(NEW_DATASET, [self.manager]), selected_idx=-1, ignore_with_ctrl=True))
        self.main_view.add_container(new_dataset_field)
        self.scatterplot_fields = {"new": new_dataset_field}

        self.bottom_view.add_leaf(Button((0.6, 0.05),  (0.3, 0.1), "compute local Rnx(K)", self.bottom_view, self.manager.uid_generator, on_value_change_listener=Listener(LAUNCH_LOCAL_RNX, [self.manager])))
        self.bottom_view.leaves[0].disable()

    def clear_hparam_container(self):
        if len(self.right_view.leaves) > 0:
            for l in self.right_view.leaves:
                if l is not None:
                    l.delete()
        self.right_view.leaves = []

    def schedule_draw(self, to_redraw, all = False, scatterplots = False, bottom = False, hparams = False):
        if all or scatterplots:
            self.main_view.schedule_draw(to_redraw)
        if all or bottom:
            self.bottom_view.schedule_draw(to_redraw)
        if all or hparams:
            self.right_view.schedule_draw(to_redraw)

    def update_dataset_tabs(self, new_tab_labels, selected_idx):
        selected_changed = False
        # rebuild tabs
        self.dataset_tabs = Mutex_choice((0,0), (1,0.1), "dataset tabs new", self.main_view, uid_generator=self.manager.uid_generator, nb_col=-1, labels = new_tab_labels, on_value_change_listener=Listener(DATASET_SELECTED, [self.manager]), selected_idx=selected_idx, ignore_with_ctrl=True, color = np.array([0, 150, 255]))
        self.main_view.change_leaf(0, self.dataset_tabs, delete_old = True)
        self.dataset_tabs.notify_on_Rclick   = True
        self.dataset_tabs.on_Rclick_listener = Listener(DELETED_DATASET, [self.manager])
        if self.selected_tab != self.dataset_tabs.value:
            selected_changed = True
            self.selected_tab = self.dataset_tabs.value
        # udate the scatterplot fields: remove any potential deleted tabs
        keys = list(self.scatterplot_fields.keys())
        for dataset_name in keys:
            if not dataset_name in new_tab_labels:
                self.scatterplot_fields[dataset_name].delete()
                del self.scatterplot_fields[dataset_name]
        # udate the scatterplot fields: add any potential additional dataset scatterplot container
        for dataset_name in new_tab_labels:
            if not dataset_name in self.scatterplot_fields:
                new_cont = Container((0,0.1), (1,0.9), dataset_name+" scatterplots", self.main_view, self.manager.uid_generator)
                new_cont.add_listener("Rclick", Listener(OPEN_ALGO_CHOICE_WINDOW, [self.manager]))
                self.scatterplot_fields[dataset_name] = new_cont

        if selected_changed:
            self.tab_change(self.selected_tab)

    def make_dataset_unclickable(self, dataset_name):
        self.scatterplot_fields["new"].leaves[0].selected_idx = -1
        self.scatterplot_fields["new"].leaves[0].disable_option(dataset_name)

    def make_dataset_clickable(self, dataset_name):
        self.scatterplot_fields["new"].leaves[0].selected_idx = -1
        self.scatterplot_fields["new"].leaves[0].enable_option(dataset_name)

    def new_tab(self, dataset_name):
        self.update_dataset_tabs([dataset_name] + [label for label in self.dataset_tabs.labels], 0)

    def delete_tab(self, dataset_name):
        self.make_dataset_clickable(dataset_name)
        prev_selected_idx  = self.dataset_tabs.selected_idx
        prev_selected_name = self.selected_tab
        new_tabs = [tab for tab in self.dataset_tabs.labels if tab != dataset_name]
        selected_idx = prev_selected_idx
        if selected_idx >= len(new_tabs):
            selected_idx = prev_selected_idx - 1
        elif new_tabs[selected_idx] != prev_selected_name:
            selected_idx = prev_selected_idx - 1
        self.update_dataset_tabs(new_tabs, selected_idx)

    def update_selected_points(self, dataset_name, point_indices, clear_drawing = False):
        if dataset_name != "new" and dataset_name in self.scatterplot_fields:
            for scatterplot in self.scatterplot_fields[dataset_name].leaves:
                scatterplot.selected_points = point_indices
                if clear_drawing:
                    scatterplot.draw_grid.fill(0)
                    scatterplot.draw_trajectory = []
                    scatterplot.drawing_start = (None, None)
                    scatterplot.last_draw     = (None, None)
        if point_indices is None:
            self.bottom_view.leaves[0].disable()
        else:
            self.bottom_view.leaves[0].enable()

    def clear_drawing(self, dataset_name):
        if dataset_name != "new" and dataset_name in self.scatterplot_fields:
            for scatterplot in self.scatterplot_fields[dataset_name].leaves:
                scatterplot.drawing = False
                scatterplot.draw_trajectory = []
                scatterplot.drawing_start   = (None, None)
                scatterplot.last_draw       = (None, None)

    def set_buttons_visible(self, dataset_name, ctrl_toggle):
        if dataset_name != "new" and dataset_name in self.scatterplot_fields:
            for scatterplot in self.scatterplot_fields[dataset_name].leaves:
                scatterplot.save_button.enable()
                scatterplot.close_button.enable()
                if ctrl_toggle:
                    scatterplot.ctrl_pressed = True

    def set_buttons_invisible(self, dataset_name, ctrl_toggle):
        if dataset_name != "new" and dataset_name in self.scatterplot_fields:
            for scatterplot in self.scatterplot_fields[dataset_name].leaves:
                scatterplot.save_button.disable()
                scatterplot.close_button.disable()
                scatterplot.draw_heatmap = False
                if ctrl_toggle:
                    scatterplot.ctrl_pressed = False

    def tab_change(self, tab_name, embeddings=None):
        self.update_selected_points(self.selected_tab, None)
        self.clear_hparam_container()
        self.dataset_tabs.set(tab_name)
        self.main_view.containers[0].disable()
        self.selected_tab = self.dataset_tabs.value
        self.main_view.change_container(0, self.scatterplot_fields[self.selected_tab], delete_old = False)
        self.scatterplot_fields[self.selected_tab].enable()
        if len(self.bottom_view.leaves) > 1:
            self.bottom_view.leaves[1].delete()
            self.bottom_view.leaves[2].delete()
            self.bottom_view.leaves = self.bottom_view.leaves[:1]
        if tab_name == "new":
            self.bottom_view.leaves[0].disable()
        else:
            general_Rnx = QA_graph((0.1, 0.17), (0.35, 0.81), 'general Rnx', self.bottom_view, self.manager.uid_generator)
            if embeddings is not None:
                for embedding_idx in range(len(self.scatterplot_fields[tab_name].leaves)):
                    scatterplot = self.scatterplot_fields[tab_name].leaves[embedding_idx]
                    embedding   = embeddings[embedding_idx]
                    if scatterplot.proj_name == embedding.proj_name:
                        if not embedding.deleted and embedding.generalQA.Rnx_ready:
                            general_Rnx.set_points(embedding.generalQA.Rnx, embedding.generalQA.Rnx_AUC, embedding.proj_name, scatterplot.color)
            local_Rnx   = QA_graph((0.55, 0.17), (0.35, 0.81), 'local Rnx', self.bottom_view, self.manager.uid_generator)
            self.bottom_view.add_leaf(general_Rnx)
            self.bottom_view.add_leaf(local_Rnx)

    def set_Rnx_max_K(self, N):
        if len(self.bottom_view.leaves) > 1:
            self.bottom_view.leaves[1].set_max_K(N-1)
            self.bottom_view.leaves[2].set_max_K(N-1)

    def update_local_Rnx(self, dataset_name, proj_name, Rnx, Rnx_AUC):
        if dataset_name in self.scatterplot_fields:
            if len(self.bottom_view.leaves) > 1:
                for scatterplot in self.scatterplot_fields[dataset_name].leaves:
                    if scatterplot.proj_name == proj_name:
                        self.bottom_view.leaves[2].set_points(Rnx, Rnx_AUC, proj_name, scatterplot.color)

    def update_general_Rnx_curve(self, dataset_name, proj_name, Rnx, Rnx_AUC):
        if dataset_name in self.scatterplot_fields:
            if len(self.bottom_view.leaves) > 1:
                for scatterplot in self.scatterplot_fields[dataset_name].leaves:
                    if scatterplot.proj_name == proj_name:
                        self.bottom_view.leaves[1].set_points(Rnx, Rnx_AUC, proj_name, scatterplot.color)

    def delete_scatterplot(self, dataset_name, proj_name):
        if not dataset_name in self.scatterplot_fields:
            return
        del_idx, found = 0, False
        for scatterplot in self.scatterplot_fields[dataset_name].leaves:
            if scatterplot.proj_name == proj_name:
                scatterplot.delete()
                found = True
                if len(self.bottom_view.leaves) > 1:
                    self.bottom_view.leaves[1].remove_projection(proj_name)
                    self.bottom_view.leaves[2].remove_projection(proj_name)
                break
            del_idx += 1
        if found:
            del self.scatterplot_fields[dataset_name].leaves[del_idx]
        # update positions of remaining scatterplots
        new_N = len(self.scatterplot_fields[dataset_name].leaves)
        if new_N > 0:
            positions, dim_pct = self.grid_values(new_N)
            for scatterplot_idx in range(len(self.scatterplot_fields[dataset_name].leaves)):
                self.scatterplot_fields[dataset_name].leaves[scatterplot_idx].update_color(self.eight_colors[scatterplot_idx])
                self.scatterplot_fields[dataset_name].leaves[scatterplot_idx].pos_pct = positions[scatterplot_idx]
                self.scatterplot_fields[dataset_name].leaves[scatterplot_idx].dim_pct = dim_pct
                self.scatterplot_fields[dataset_name].leaves[scatterplot_idx].update_pos_and_dim()
                scatterplot_proj_name = self.scatterplot_fields[dataset_name].leaves[scatterplot_idx].proj_name
                # rnx graph things
                if len(self.bottom_view.leaves) > 1:
                    if scatterplot_proj_name in self.bottom_view.leaves[1].auc_texts:
                        self.bottom_view.leaves[1].auc_texts[scatterplot_proj_name].color = self.eight_colors[scatterplot_idx]
                        self.bottom_view.leaves[1].recompute_drawing(color_change = True)
                        if scatterplot_proj_name in self.bottom_view.leaves[2].auc_texts:
                            self.bottom_view.leaves[2].auc_texts[scatterplot_proj_name].color = self.eight_colors[scatterplot_idx]
                            self.bottom_view.leaves[2].recompute_drawing(color_change = True)
                    else:
                        self.manager.ask_redraw(self.main_view)
                        self.manager.ask_redraw(self.right_view)
                        self.manager.ask_redraw(self.bottom_view)

    def scatterplot_right_click(self, mouse_pos, windows, curr_K, max_K):
        w, h = (200, 100)
        window_abspos = (mouse_pos[0], max(self.scatterplot_fields[self.selected_tab].abs_pos[1], mouse_pos[1]-h))
        right_click_window = Window(window_abspos, (w, h), "K choice", True, self.manager.uid_generator, color=self.color, background_color=self.background_color, close_on_notify=True)
        listener = Listener(SELECTION_K_CHOICE, [self.manager, right_click_window])
        right_click_window.add_leaf(Number_selector((0,0), (1, 1), "selection K", right_click_window, uid_generator=self.manager.uid_generator, nb_type="int", min_val=1, max_val=max_K, step=1, init_val=curr_K, default=curr_K, on_value_change_listener=listener))
        right_click_window.open(abs_pos = window_abspos, windows_list = windows)

    def open_algo_choice_window(self, mouse_pos, windows, algo_list, dataset_name):
        self.clear_hparam_container()
        w, h = (200, 300)
        window_abspos = (mouse_pos[0], max(self.scatterplot_fields[self.selected_tab].abs_pos[1], mouse_pos[1]-h))
        right_click_window = Window(window_abspos, (w, h), "algo choice", True, self.manager.uid_generator, color=self.color, background_color=self.background_color, close_on_notify=True)
        listener = Listener(ALGO_CHOICE_DONE, [self.manager, right_click_window])
        listener.misc_info = dataset_name
        right_click_window.add_leaf(Mutex_choice((0,0), (1,1), "algo choice mutex", right_click_window, uid_generator=self.manager.uid_generator, nb_col=1, labels = algo_list, on_value_change_listener=listener, selected_idx=-1, ignore_with_ctrl=True))
        right_click_window.open(abs_pos = window_abspos, windows_list = windows)

    def build_hparams_selector(self, hparam_schematics, algorithm_in_construction_already_exists=False, dataset_name=None, proj_name=None):
        self.clear_hparam_container()
        done_listener = Listener(HPARAM_CHOICE_DONE, [self.manager])
        done_listener.misc_info = (algorithm_in_construction_already_exists, dataset_name, proj_name)
        ok_button = Button((0.2, 0.92),  (0.6, 0.07), "OK", self.right_view, self.manager.uid_generator, on_value_change_listener=done_listener)
        scrollable = Scrollable_bundle((0.01, 0), (0.98, 0.9), "hparams", self.right_view, self.manager.uid_generator)
        items = []
        for key in hparam_schematics:
            scheme = hparam_schematics[key]
            if scheme[0] in ["float", "int", "int-str", "float-str"]:
                sel = Number_selector((0,0), (1, 0.1), key, scrollable, uid_generator=self.manager.uid_generator, nb_type=scheme[0], min_val=scheme[1], max_val=scheme[2], step=scheme[3],init_val=scheme[4], default=scheme[4])
            elif scheme[0] == "bool":
                idx = 1
                if scheme[2] == "True" or scheme[2] == True:
                    idx = 0
                sel = Mutex_with_title((0,0), (1, 0.1), key, scrollable, uid_generator=self.manager.uid_generator, nb_col=2, labels=["True","False"], selected_idx=idx)
            else: # string
                sel = String_selector((0,0), (1, 0.1), key, scrollable, uid_generator=self.manager.uid_generator, values=scheme[2], default_value=scheme[3])
            items.append(sel)
        scrollable.set_items(items, new_title = "Hyperparameters")
        self.right_view.add_leaf(scrollable)
        self.right_view.add_leaf(ok_button)

    def read_hparams_selector(self):
        raw = self.right_view.leaves[0].get_values()
        sanitized = {}
        for key in raw:
            v = raw[key]
            if raw[key] in ["True", "true"]:
                v = True
            elif raw[key] in ["False", "false"]:
                v = False
            sanitized[key] = v
        self.clear_hparam_container()
        return sanitized

    def get_scatterplot(self, dataset_name, proj_name):
        if not dataset_name in self.scatterplot_fields or dataset_name != self.selected_tab:
            return
        for scatterplot in self.scatterplot_fields[dataset_name].leaves:
            if scatterplot.proj_name == proj_name:
                return scatterplot

    def selected_scatterplot_changed(self, dataset_name, proj_name, algorithm_in_construction, algorithm_in_construction_already_exists):
        if not dataset_name in self.scatterplot_fields or dataset_name != self.selected_tab:
            return
        for scatterplot in self.scatterplot_fields[dataset_name].leaves:
            if scatterplot.proj_name == proj_name:
                scatterplot.selected = True
            else:
                scatterplot.selected = False
        if algorithm_in_construction is None: # unselected all
            return
        schematics = algorithm_in_construction.get_hyperparameter_schematics_copy()
        hparams    = algorithm_in_construction.get_hyperparameters()
        for key in schematics:
            schematics[key][-1] = hparams[key]
        self.build_hparams_selector(schematics, algorithm_in_construction_already_exists, dataset_name, proj_name)

    def get_selected_scatterplot_name(self):
        if self.selected_tab != "new":
            for scatterplot in self.scatterplot_fields[self.selected_tab].leaves:
                if scatterplot.selected:
                    return scatterplot.proj_name
        return None


    def grid_values(self, N):
        if N == 1:
            return [[0, 0]], (1, 1)
        elif N == 2:
            return [[0, 0], [0.5, 0]], (0.5, 1)
        elif N == 3:
            return [[0, 0], [0.33, 0], [0.66, 0]], (0.33, 1)
        elif N == 4:
            return [[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]], (0.5, 0.5)
        elif N == 5:
            return [[0, 0], [0.33, 0], [0.66, 0], [0, 0.5], [0.33, 0.5]], (0.33, 0.5)
        elif N == 6:
            return [[0, 0], [0.33, 0], [0.66, 0], [0, 0.5], [0.33, 0.5], [0.66, 0.5]], (0.33, 0.5)
        elif N == 7:
            return [[0, 0], [0.25, 0], [0.5, 0], [0.75, 0], [0, 0.5], [0.25, 0.5], [0.5, 0.5]], (0.25, 0.5)
        else:
            return [[0, 0], [0.25, 0], [0.5, 0], [0.75, 0], [0, 0.5], [0.25, 0.5], [0.5, 0.5], [0.75, 0.5]], (0.25, 0.5)

    def reset_scatterplot(self, dataset_name, proj_name):
        if not dataset_name in self.scatterplot_fields:
            return
        for scatterplot in self.scatterplot_fields[dataset_name].leaves:
            if scatterplot.proj_name == proj_name:
                scatterplot.converged = False

    def init_new_scatterplot(self, dataset_name, proj_name, M, is_labeled, is_classification, n_scatterplot_diff = 0):
        updated_nb_scatterplots = len(self.scatterplot_fields[dataset_name].leaves) + n_scatterplot_diff
        if updated_nb_scatterplots > 8:
            return False
        positions, dim_pct = self.grid_values(updated_nb_scatterplots)
        for scatterplot_idx in range(len(self.scatterplot_fields[dataset_name].leaves)):
            self.scatterplot_fields[dataset_name].leaves[scatterplot_idx].pos_pct = positions[scatterplot_idx]
            self.scatterplot_fields[dataset_name].leaves[scatterplot_idx].dim_pct = dim_pct
            self.scatterplot_fields[dataset_name].leaves[scatterplot_idx].update_pos_and_dim()

        scatterplot = Scatterplot(positions[-1], dim_pct, dataset_name, proj_name, self.scatterplot_fields[dataset_name], uid_generator=self.manager.uid_generator, is_labeled=is_labeled, is_classification=is_classification,\
                    manager=self.manager, color = self.eight_colors[updated_nb_scatterplots-1])
        scatterplot.update_pos_and_dim()
        heatmap = Heatmap((0,0), (0.35,0.35), proj_name , scatterplot, M, self.manager.uid_generator)
        scatterplot.heatmap = heatmap
        self.scatterplot_fields[dataset_name].add_leaf(scatterplot)
        return True

    def scatterplot_point_update(self, dataset_name, proj_name, X_LD, Y, Y_colors, converged):
        if dataset_name in self.scatterplot_fields:
            for scatterplot in self.scatterplot_fields[dataset_name].leaves:
                if scatterplot.proj_name == proj_name:
                    scatterplot.set_points(X_LD, Y, Y_colors)
                    if converged:
                        scatterplot.converged = True
                        return
                    else:
                        return scatterplot

    def ctrl_press(self, dataset_name):
        self.clear_hparam_container()
        self.update_selected_points(dataset_name, None)
        self.clear_drawing(dataset_name)
        self.set_buttons_visible(dataset_name, ctrl_toggle = True)

    def ctrl_unpress(self, dataset_name):
        self.set_buttons_invisible(dataset_name, ctrl_toggle = True)
