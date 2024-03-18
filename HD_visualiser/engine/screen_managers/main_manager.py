from engine.screen_managers.manager import Manager
from engine.gui.listener import Listener
from DR_algorithms.DR_algorithm import DR_algorithm, get_DR_algorithm
from engine.gui.event_ids import *
from QA.general_QA import local_rnx_auc
from utils import get_dataset_names, random_colors, get_algorithm_names
from Data_loader import Dataset_loader
from engine.Dataset import Dataset
from engine.Embedding import Embedding
import threading
from sklearn.metrics import pairwise_distances
import numpy as np
from engine.screens.main_screen import Main_screen

class Main_manager(Manager):
    def __init__(self, config, main_window, theme, uid_generator, absQA_manager, relQA_manager):
        super(Main_manager, self).__init__("main", initial_state=True)
        self.type_identifier = "main manger"
        self.regr_color_using_span = config["regr color with span"]
        self.absQA_manager = absQA_manager
        self.absQA_manager.active = False
        self.relQA_manager = relQA_manager
        self.relQA_manager.active = False
        self.active  = True
        self.deleted = False
        self.theme = theme
        self.main_window        = main_window
        self.main_window.awaiting_key_press.append(self)
        self.uid_generator      = uid_generator
        self.data_loader        = Dataset_loader(config)
        self.dataset_names      = get_dataset_names()
        self.DR_algo_names      = get_algorithm_names()
        self.screen             = Main_screen(theme, main_window, self)

        self.open_datasets             = {"new": None}
        self.open_datasets_Ycolors     = {"new": np.array([np.array([50, 0, 200])])}
        self.open_datasets_Projections = {"new": []}
        self.open_datasets_models      = {"new": []}
        self.current_dataset           = "new"

        self.selected_points = None
        self.ctrl_pressed    = False
        self.selection_K       = 30
        self.selection_max_K   = 300

        self.algorithm_in_construction                = None
        self.algorithm_in_construction_already_exists = False # changing an exisiting scatterplot
        self.lock = threading.Lock()


    #   ~~~~~~~~~~~~   towards other managers    ~~~~~~~~~~~~~~~~~~
    # def propagate_notification_to_other_manager(self, event_class, value):
    #     print("propagate to other manager")

    def wake_up(self, prev_manager):
        super(Main_manager, self).wake_up(prev_manager)
        if prev_manager.type_identifier == "absQA_manager" and prev_manager.selected_proj_name is not None and self.current_dataset != "new":
            self.selected_changed(self.current_dataset, prev_manager.selected_proj_name)
        if prev_manager.type_identifier == "relQA_manager" and prev_manager.selected_embedding_L_name is not None and self.current_dataset != "new":
            self.selected_changed(self.current_dataset, prev_manager.selected_embedding_L_name)

    def local_QA_done(self, dataset_name, proj_name, QA_type, to_redraw):
        if dataset_name != self.current_dataset or not self.open_datasets[dataset_name].initialised:
            return
        dataset = self.open_datasets[dataset_name]
        embedding = None
        for proj in self.open_datasets_Projections[dataset_name]:
            if proj.proj_name == proj_name:
                embedding = proj
        if embedding is None:
            return
        if QA_type == "dist":
            if self.absQA_manager.active:
                self.absQA_manager.receive_local_distQA(dataset_name, proj_name, embedding.local_distQA, to_redraw=to_redraw)
            elif self.relQA_manager.active:
                self.relQA_manager.receive_local_distQA(dataset_name, proj_name, embedding.local_distQA, to_redraw=to_redraw)
        elif QA_type == "label":
            if self.absQA_manager.active:
                self.absQA_manager.receive_local_labelQA(dataset_name, proj_name, embedding.local_labelQA, to_redraw=to_redraw)
            elif self.relQA_manager.active:
                self.relQA_manager.receive_local_labelQA(dataset_name, proj_name, embedding.local_labelQA, to_redraw=to_redraw)
            # if dataset_name in self.open_datasets:
            #     for embedding in self.open_datasets_Projections[dataset_name]:
            #         if embedding.proj_name ==  proj_name:
            #             embedding.D = None
            #             embedding.neighbours = embedding.neighbours[:, :self.selection_max_K]
        elif QA_type == "neighbour":
            if self.absQA_manager.active:
                self.absQA_manager.receive_local_neighQA(dataset_name, proj_name, embedding.local_neighQA, to_redraw=to_redraw)
            elif self.relQA_manager.active:
                self.relQA_manager.receive_local_neighQA(dataset_name, proj_name, embedding.local_neighQA, to_redraw=to_redraw)

    def notify_other_screen_proj_done(self):
        if self.current_dataset != "new":
            if self.absQA_manager.active:
                self.absQA_manager.rebuild_all(self.current_dataset, first_rebuild=False)
            elif self.relQA_manager.active:
                self.relQA_manager.rebuild_all(self.current_dataset, first_rebuild=False)

    def shepard_ready(self, dataset_name, proj_name, embedding_D, to_redraw):
        if dataset_name != self.current_dataset or not self.open_datasets[dataset_name].initialised:
            return
        dataset_D   = self.open_datasets[dataset_name].D
        dist_idxes1  = self.open_datasets[dataset_name].shepard_distances1
        dist_idxes2  = self.open_datasets[dataset_name].shepard_distances2
        if self.absQA_manager.active:
            self.absQA_manager.receive_shepard_data(dataset_name, proj_name, embedding_D, dataset_D, dist_idxes1, dist_idxes2, to_redraw=to_redraw)
        elif self.relQA_manager.active:
            self.relQA_manager.receive_shepard_data(dataset_name, proj_name, embedding_D, dataset_D, dist_idxes1, dist_idxes2, to_redraw=to_redraw)


    def Dcorr_ready(self, dataset_name, proj_name, Dcorr, to_redraw):
        if dataset_name != self.current_dataset or not self.open_datasets[dataset_name].initialised:
            return
        if self.absQA_manager.active:
            self.absQA_manager.receive_Dcorr_data(dataset_name, proj_name, Dcorr, to_redraw=to_redraw)
        elif self.relQA_manager.active:
            self.relQA_manager.receive_Dcorr_data(dataset_name, proj_name, Dcorr, to_redraw=to_redraw)

    def Rnx_ready(self, dataset_name, proj_name, Rnx, Rnx_AUC, to_redraw):
        if dataset_name != self.current_dataset or not self.open_datasets[dataset_name].initialised:
            return
        if self.absQA_manager.active:
            self.absQA_manager.receive_Rnx_data(dataset_name, proj_name, Rnx, Rnx_AUC, to_redraw=to_redraw)
        elif self.relQA_manager.active:
            self.relQA_manager.receive_Rnx_data(dataset_name, proj_name, Rnx, Rnx_AUC, to_redraw=to_redraw)

    def KNNgain_ready(self, dataset_name, proj_name, KNNgain, KNNgain_AUC, to_redraw):
        if dataset_name != self.current_dataset or not self.open_datasets[dataset_name].initialised:
            return
        if self.absQA_manager.active:
            self.absQA_manager.receive_KNNgain_data(dataset_name, proj_name, KNNgain, KNNgain_AUC, to_redraw=to_redraw)
        elif self.relQA_manager.active:
            self.relQA_manager.receive_KNNgain_data(dataset_name, proj_name, KNNgain, KNNgain_AUC, to_redraw=to_redraw)
    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_selected_embedding_name(self):
        if self.current_dataset == "new":
            return None
        return self.screen.get_selected_scatterplot_name()

    def get_selected_embedding(self):
        selected_name = self.get_selected_embedding_name()
        if selected_name is not None:
            for embedding in self.open_datasets_Projections[self.current_dataset]:
                if embedding.proj_name == selected_name:
                    return embedding
        return None

    def local_Rnx_done(self, dataset_name, proj_name, rnxk, auc):
        if dataset_name in self.open_datasets and dataset_name != "new":
            self.screen.update_local_Rnx(dataset_name, proj_name, rnxk, auc)

    def launch_local_Rnx(self):
        if self.current_dataset == "new" or self.selected_points is None:
            return
        if self.current_dataset in self.open_datasets:
            dataset = self.open_datasets[self.current_dataset]
            if not dataset.initialised:
                print("cannot launch local Rnx: dataset still initialising")
                return
            perms_hd = dataset.neighbours
            D_HD     = dataset.D
            for embedding in self.open_datasets_Projections[self.current_dataset]:
                if embedding.neighbours_ready and not embedding.deleted:
                    perms_ld = embedding.neighbours
                    D_LD = pairwise_distances(embedding.X_LD)
                    listener = Listener(LOCAL_RNX_DONE, [self])
                    local_Rnx_thread = threading.Thread(target=local_rnx_auc, args=[self.current_dataset, embedding.proj_name, perms_hd, perms_ld, D_HD, D_LD, self.selected_points, listener])
                    local_Rnx_thread.start()

    def set_Rnx_max_K(self):
        if self.current_dataset == "new":
            return
        if self.current_dataset in self.open_datasets:
            dataset = self.open_datasets[self.current_dataset]
            self.screen.set_Rnx_max_K(dataset.X.shape[0])

    def update_heatmap(self, LD_location, dataset_name, proj_name, mouse_pos, to_redraw):
        if dataset_name != self.current_dataset:
            return
        if self.open_datasets[dataset_name].is_dists:
            return
        for embedding in self.open_datasets_Projections[dataset_name]:
            if embedding.proj_name == proj_name:
                selected_point = embedding.find_K_closest_to_cursor(LD_location, 1)
                dataset = self.open_datasets[dataset_name]
                mins, maxes = dataset.X_mins, dataset.X_maxes
                W = (maxes - mins) + 1e-9
                intensities = (dataset.X[selected_point] - mins) / W
                if dataset.is_labeled:
                    label = dataset.Y[selected_point]
                else:
                    label = 0
                scatterplot = self.screen.get_scatterplot(dataset_name, proj_name)
                if not scatterplot.deleted:
                    scatterplot.heatmap.update_values(intensities.ravel(), label)
                    scatterplot.heatmap.update_pos_and_dim(mouse_pos)
                    scatterplot.schedule_draw(to_redraw)

    def general_QA_done(self, dataset_name, proj_name, curve, auc, QA_type, to_redraw):
        if not dataset_name in self.open_datasets_Projections:
            return
        for proj in self.open_datasets_Projections[dataset_name]:
            if proj.proj_name == proj_name:
                if QA_type == "Rnx":
                    proj.generalQA.Rnx       = curve
                    proj.generalQA.Rnx_AUC   = np.round(auc, 3)
                    proj.generalQA.Rnx_ready = True
                    if dataset_name == self.current_dataset:
                        self.screen.update_general_Rnx_curve(dataset_name, proj_name, curve, np.round(auc, 3))
                        if not self.active:
                            self.Rnx_ready(dataset_name, proj_name, curve, np.round(auc, 3), to_redraw)
                elif QA_type == "Dcorr":
                    proj.generalQA.Dcorr     = np.round(auc, 3)
                    proj.generalQA.Dcorr_ready = True
                    if dataset_name == self.current_dataset:
                        if not self.active:
                            self.Dcorr_ready(dataset_name, proj_name, np.round(auc, 3), to_redraw=to_redraw)
                else:
                    proj.generalQA.KNNgain       = curve
                    proj.generalQA.KNNgain_AUC   = np.round(auc, 3)
                    proj.generalQA.KNNgain_ready = True
                    if dataset_name == self.current_dataset:
                        if not self.active:
                            self.KNNgain_ready(dataset_name, proj_name, curve, np.round(auc, 3), to_redraw=to_redraw)

    def KNN_selection(self, LD_location, dataset_name, proj_name):
        if dataset_name != self.current_dataset:
            return
        for embedding in self.open_datasets_Projections[dataset_name]:
            if embedding.proj_name == proj_name:
                self.selected_points = embedding.find_K_closest_to_cursor(LD_location, self.selection_K)
                self.screen.update_selected_points(dataset_name, self.selected_points)

    def draw_selection(self, selected_points, dataset_name, proj_name):
        if dataset_name != self.current_dataset:
            return
        self.selected_points = selected_points
        self.screen.update_selected_points(dataset_name, self.selected_points, clear_drawing=True)

    def selected_changed(self, dataset_name, proj_name):
        if dataset_name != self.current_dataset:
            return
        proj_idx = 0; found = False
        for proj in self.open_datasets_Projections[dataset_name]:
            if proj.proj_name == proj_name:
                found = True
                break
            proj_idx += 1
        if not found:
            return
        self.algorithm_in_construction = self.open_datasets_models[dataset_name][proj_idx]
        self.algorithm_in_construction_already_exists = True
        self.screen.selected_scatterplot_changed(dataset_name, proj_name, self.algorithm_in_construction, self.algorithm_in_construction_already_exists)

    def projection_done(self, dataset_name, proj_name, X_LD):
        if dataset_name in self.open_datasets:
            for embedding in self.open_datasets_Projections[dataset_name]:
                if embedding.proj_name ==  proj_name:
                    embedding.X_LD = X_LD
                    self.screen.scatterplot_point_update(dataset_name, proj_name, X_LD, self.open_datasets[dataset_name].Y, self.open_datasets_Ycolors[dataset_name], converged=True)
                    embedding.done = True
                    if not self.active and self.current_dataset == dataset_name:
                        self.notify_other_screen_proj_done()

    def save_projection(self, value):
        dataset_name, proj_name = value.split("___")
        if dataset_name in self.open_datasets:
            for embedding in self.open_datasets_Projections[dataset_name]:
                if embedding.proj_name ==  proj_name and embedding.done:
                    try:
                        embedding.save("./saved_embeddings/"+value+".npy", self.open_datasets[dataset_name])
                    except Exception as e:
                        print("error when saving projection: ", e)

    def close_projection(self, value):
        if not self.active:
            return
        dataset_name, proj_name = value.split("___")
        if dataset_name in self.open_datasets:
            del_idx, found = 0, False
            for embedding in self.open_datasets_Projections[dataset_name]:
                if embedding.proj_name ==  proj_name:
                    embedding.delete()
                    found = True
                    break
                del_idx += 1
            if found:
                self.open_datasets_Projections[dataset_name][del_idx].delete()
                del self.open_datasets_Projections[dataset_name][del_idx]
                self.open_datasets_models[dataset_name][del_idx].deleted = True
                del self.open_datasets_models[dataset_name][del_idx]
        self.absQA_manager.closed_projection(proj_name)
        self.relQA_manager.closed_projection(proj_name)
        self.screen.delete_scatterplot(dataset_name, proj_name)

    def ctrl_press_start(self):
        self.ctrl_pressed = True
        self.selected_points = None
        self.screen.ctrl_press(self.current_dataset)

    def ctrl_press_end(self):
        self.ctrl_pressed = False
        self.screen.ctrl_unpress(self.current_dataset)

    def scatterplot_KNN_computed(self, dataset_name, proj_name, tree, neighbours, D, to_redraw):
        if not dataset_name in self.open_datasets:
            return
        for embedding in self.open_datasets_Projections[dataset_name]:
            if embedding.proj_name == proj_name:
                embedding.KD_tree          = tree
                embedding.neighbours       = neighbours
                embedding.D                = D
                embedding.neighbours_ready = True
                if not self.active:
                    self.shepard_ready(dataset_name, proj_name, embedding.D, to_redraw)
        if self.selected_points is not None:
            scatterplot = self.screen.get_scatterplot(dataset_name, proj_name)
            if scatterplot is not None:
                scatterplot.selected_points = self.selected_points
                if scatterplot is not None and dataset_name == self.current_dataset:
                    self.ask_redraw(scatterplot)


    def relaunch_projection(self, dataset_name, proj_name):
        if self.algorithm_in_construction is None:
            return
        if not dataset_name in self.open_datasets_Projections:
            return
        self.screen.reset_scatterplot(dataset_name, proj_name)
        new_embedding = Embedding(dataset_name=self.current_dataset, projection_name=proj_name, N=self.open_datasets[self.current_dataset].X.shape[0], is_dists=self.open_datasets[self.current_dataset].is_dists)
        for embedding_idx in range(len(self.open_datasets_Projections[dataset_name])):
            if self.open_datasets_Projections[dataset_name][embedding_idx].proj_name == proj_name:
                self.open_datasets_Projections[dataset_name][embedding_idx].delete()
                self.open_datasets_Projections[dataset_name][embedding_idx] = new_embedding
        new_embedding.start(dataset=self.open_datasets[dataset_name], model=self.algorithm_in_construction, manager=self)

    def launch_projection(self):
        if self.algorithm_in_construction is None:
            return
        base_algo_name = self.algorithm_in_construction.name
        algo_number = 1
        for proj in self.open_datasets_Projections[self.current_dataset]:
            if base_algo_name in proj.name:
                algo_number = max(algo_number, int(proj.name.split('#')[1]))+1
        proj_name = base_algo_name + '#' + str(algo_number)
        ret = self.screen.init_new_scatterplot(self.current_dataset, proj_name, self.open_datasets[self.current_dataset].X.shape[1], is_labeled=self.open_datasets[self.current_dataset].is_labeled, is_classification=self.open_datasets[self.current_dataset].is_classification, n_scatterplot_diff=1)
        if not ret:
            return
        embedding = Embedding(dataset_name=self.current_dataset, projection_name=proj_name, N=self.open_datasets[self.current_dataset].X.shape[0], is_dists=self.open_datasets[self.current_dataset].is_dists)
        self.open_datasets_Projections[self.current_dataset].append(embedding)
        self.open_datasets_models[self.current_dataset].append(self.algorithm_in_construction)
        self.algorithm_in_construction.tag(self.current_dataset, proj_name)
        embedding.start(dataset=self.open_datasets[self.current_dataset], model=self.algorithm_in_construction, manager=self)

    def get_notified(self, event_class, id, value, to_redraw = []):
        with self.lock:

            if event_class == DATASET_SELECTED and not self.ctrl_pressed:
                self.change_dataset(value)
                self.set_Rnx_max_K()
                # self.screen.schedule_draw(to_redraw, all = True)
                self.ask_redraw(self.screen.main_view)
                self.ask_redraw(self.screen.bottom_view)
                self.ask_redraw(self.screen.right_view)

            elif event_class == NEW_DATASET and not self.ctrl_pressed: # clicked in dataset selection screen
                if value in self.open_datasets:
                    self.change_dataset(value)
                else:
                    self.open_new_full_dataset(value)
                self.screen.scatterplot_fields["new"].leaves[0].unselect_all()
                self.set_Rnx_max_K()
                if self.active:
                    self.screen.schedule_draw(to_redraw, all = True)

            elif event_class == DELETED_DATASET and not self.ctrl_pressed:
                self.close_dataset(value)
                if self.active:
                    self.screen.schedule_draw(to_redraw, all = True)

            elif event_class == OPEN_ALGO_CHOICE_WINDOW and not self.ctrl_pressed:
                if len(self.open_datasets_Projections[self.current_dataset]) < 8:
                    dataset_name, windows, mouse_pos = value
                    self.screen.open_algo_choice_window(mouse_pos, windows, self.DR_algo_names, dataset_name)

            elif event_class == ALGO_CHOICE_DONE and not self.ctrl_pressed:
                if len(self.open_datasets_Projections[self.current_dataset]) < 8:
                    algorithm_name, dataset = value
                    self.algorithm_in_construction = get_DR_algorithm(algorithm_name)
                    self.algorithm_in_construction_already_exists = False
                    self.screen.build_hparams_selector(self.algorithm_in_construction.get_hyperparameter_schematics())

            elif event_class == HPARAM_CHOICE_DONE and not self.ctrl_pressed:
                algorithm_in_construction_already_exists, dataset_name, proj_name = value[1]
                if len(self.open_datasets_Projections[self.current_dataset]) < 8 or algorithm_in_construction_already_exists:
                    hparams = self.screen.read_hparams_selector()
                    if hparams is not None and self.algorithm_in_construction is not None:
                        self.algorithm_in_construction.set_hparams(hparams)
                        if self.current_dataset != "new" and self.open_datasets[self.current_dataset] is not None:
                            if not algorithm_in_construction_already_exists:
                                self.launch_projection()
                            else:
                                self.relaunch_projection(dataset_name, proj_name)
                    self.screen.selected_scatterplot_changed(self.current_dataset, "None", None, False) # clear selected
                    if self.active:
                        self.screen.schedule_draw(to_redraw, hparams = True, scatterplots = True)

            elif event_class == CTRL_KEY_CHANGE:
                if value[0]:
                    self.ctrl_press_start()
                    self.screen.selected_scatterplot_changed(self.current_dataset, "None", None, False) # clear selected
                else:
                    self.ctrl_press_end()
                if self.active:
                    self.screen.schedule_draw(to_redraw, all = True)

            elif event_class == SCATTERPLOT_CLOSED:
                self.close_projection(value[1])
                if self.active:
                    self.screen.schedule_draw(to_redraw, all = True)

            elif event_class == SCATTERPLOT_SAVE:
                self.save_projection(value[1])

            elif event_class == PROJECTION_ERROR:
                dataset_name, proj_name = value.split(" ~ ")
                self.close_projection(dataset_name+"___"+proj_name)
                self.ask_redraw(self.screen.main_view)

            elif event_class == PROJECTION_DONE:
                dataset_name, proj_name = value[0].split(" ~ ")
                self.projection_done(dataset_name, proj_name, value[1])
                self.ask_redraw(self.screen.main_view)

            elif event_class == SCATTERPLOT_SELECTED:
                self.selected_changed(value[0], value[1])
                if self.active:
                    self.screen.schedule_draw(to_redraw, scatterplots = True, hparams = True)

            elif event_class == EMBEDDING_KNN_DONE:
                embedding_ID, tree, neighbours, D_LD = value
                dataset_name, proj_name = embedding_ID.split(" ~ ")
                self.scatterplot_KNN_computed(dataset_name, proj_name, tree, neighbours, D_LD, to_redraw)
                scatterplot = self.screen.get_scatterplot(dataset_name, proj_name)
                if scatterplot is not None:
                    self.ask_redraw(scatterplot)

            elif event_class == SCATTERPLOT_KNN_SELECTION:
                LD_location, dataset_name, proj_name = value
                self.KNN_selection(LD_location, dataset_name, proj_name)
                if self.active:
                    self.screen.schedule_draw(to_redraw, scatterplots = True, bottom = True)

            elif event_class == SCATTERPLOT_DRAW_SELECTION:
                selected_points, dataset_name, proj_name = value
                self.draw_selection(selected_points, dataset_name, proj_name)
                if self.active:
                    self.screen.schedule_draw(to_redraw, scatterplots = True, bottom = True)

            elif event_class == SCATTERPLOT_RIGHT_CLICK:
                self.screen.scatterplot_right_click(value[0], value[1], self.selection_K, self.selection_max_K)

            elif event_class == SELECTION_K_CHOICE:
                self.selection_K = min(self.selection_max_K, value)

            elif event_class == UPDATE_HEATMAP:
                mouse_pos, LD_location, dataset_name, proj_name = value
                self.update_heatmap(LD_location, dataset_name, proj_name, mouse_pos, to_redraw)

            elif event_class == GENERAL_RNX:
                embedding_ID, rnx, auc = value
                dataset_name, proj_name = embedding_ID.split(" ~ ")
                self.general_QA_done(dataset_name, proj_name, rnx, auc, QA_type = "Rnx", to_redraw=to_redraw)
                self.ask_redraw(self.screen.bottom_view)

            elif event_class == KNN_GAIN_DONE:
                embedding_ID, KNNgain, KNNgain_auc = value
                dataset_name, proj_name = embedding_ID.split(" ~ ")
                self.general_QA_done(dataset_name, proj_name, KNNgain, KNNgain_auc, QA_type = "KNNgain", to_redraw=to_redraw)

            elif event_class == DCORR_DONE:
                embedding_ID, Dcorr = value
                dataset_name, proj_name = embedding_ID.split(" ~ ")
                self.general_QA_done(dataset_name, proj_name, None, Dcorr, QA_type = "Dcorr", to_redraw=to_redraw)


            elif event_class == LAUNCH_LOCAL_RNX:
                self.launch_local_Rnx()

            elif event_class == LOCAL_RNX_DONE:
                dataset_name, proj_name, rnxk, auc = value
                self.local_Rnx_done(dataset_name, proj_name, rnxk, auc)
                self.ask_redraw(self.screen.bottom_view)

            elif event_class == CONVERGENCE_UPDATE:
                dataset_name, proj_name, X_LD, model = value
                if not model.deleted:
                    self.convergence_update(dataset_name, proj_name, X_LD)

            elif event_class == LOCAL_DISTQA:
                dataset_name, proj_name = value.split(" ~ ")
                self.local_QA_done(dataset_name, proj_name, "dist", to_redraw)

            elif event_class == LOCAL_NEIGHQA:
                dataset_name, proj_name = value.split(" ~ ")
                self.local_QA_done(dataset_name, proj_name, "neighbour", to_redraw)

            elif event_class == LOCAL_LABELQA:
                dataset_name, proj_name = value.split(" ~ ")
                self.local_QA_done(dataset_name, proj_name, "label", to_redraw)

            # if not self.active:
            #     self.propagate_notification_to_other_manager(event_class, value)



    def convergence_update(self, dataset_name, proj_name, X_LD):
        if dataset_name not in self.open_datasets:
            return
        for proj in self.open_datasets_Projections[dataset_name]:
            if proj.proj_name == proj_name:
                Y        = self.open_datasets[dataset_name].Y
                Y_colors = self.open_datasets_Ycolors[dataset_name]
                scatterplot = self.screen.scatterplot_point_update(dataset_name, proj_name, X_LD, Y, Y_colors, converged=False)
                if scatterplot is not None and dataset_name == self.current_dataset:
                    self.ask_redraw(scatterplot)

    def change_dataset(self, dataset_name):
        self.absQA_manager.selected_proj_name = None
        self.absQA_manager.selected_embedding = None
        self.relQA_manager.set_selected(None, None, L = True)
        self.relQA_manager.set_selected(None, None, L = False)
        self.selected_points = None
        self.algorithm_in_construction = None
        self.algorithm_in_construction_already_exists = False
        self.current_dataset = dataset_name
        if dataset_name != "new":
            dataset_max_K = min(int(self.open_datasets[dataset_name].X.shape[0]*0.5), 300)
            self.selection_K       = min(self.selection_K,  dataset_max_K)
            self.selection_max_K   = min(300, dataset_max_K)
            self.screen.tab_change(dataset_name, self.open_datasets_Projections[dataset_name])
        else:
            self.screen.tab_change(dataset_name)

    def close_dataset(self, dataset_name):
        if dataset_name == "new":
            return
        if self.current_dataset == dataset_name:
            self.change_dataset("new")
        # close in Main_screen
        self.screen.delete_tab(dataset_name)
        # close here
        self.open_datasets[dataset_name].delete()
        del self.open_datasets[dataset_name]
        del self.open_datasets_Ycolors[dataset_name]
        for proj in self.open_datasets_Projections[dataset_name]:
            if proj is not None:
                proj.delete()
        del self.open_datasets_Projections[dataset_name]
        del self.open_datasets_models[dataset_name]

    def open_new_subset_dataset(self, to_redraw):
        with self.lock:
            if (self.current_dataset == "new") or (self.current_dataset not in self.open_datasets) or (self.selected_points is None) or (len(self.selected_points) < 50):
                return
            curr_dataset = self.open_datasets[self.current_dataset]
            if not curr_dataset.is_dists:
                if curr_dataset.is_labeled:
                    X, Y, is_labeled, is_classification, colors = curr_dataset.X[self.selected_points], curr_dataset.Y[self.selected_points], curr_dataset.is_labeled, curr_dataset.is_classification, self.open_datasets_Ycolors[self.current_dataset][self.selected_points]
                else:
                    X, Y, is_labeled, is_classification, colors = curr_dataset.X[self.selected_points], None, curr_dataset.is_labeled, curr_dataset.is_classification, self.open_datasets_Ycolors[self.current_dataset][self.selected_points]
            else:
                if curr_dataset.is_labeled:
                    X, Y, is_labeled, is_classification, colors = curr_dataset.D[self.selected_points][:,self.selected_points], curr_dataset.Y[self.selected_points], curr_dataset.is_labeled, curr_dataset.is_classification, self.open_datasets_Ycolors[self.current_dataset][self.selected_points]
                else:
                    X, Y, is_labeled, is_classification, colors = curr_dataset.D[self.selected_points][:,self.selected_points], None, curr_dataset.is_labeled, curr_dataset.is_classification, self.open_datasets_Ycolors[self.current_dataset][self.selected_points]


            highest_nb = 0
            base_name = self.current_dataset.split(" :: ")[0]
            for key in self.open_datasets:
                if base_name in key:
                    name_split = key.split(" :: ")
                    if len(name_split) > 1:
                        int_value = int(name_split[1])
                        if int_value > highest_nb:
                            highest_nb = int_value
            dataset_name = base_name + " :: " + str(highest_nb+1)
            self.new_dataset(dataset_name, X, Y, is_labeled, is_classification, colors, curr_dataset.is_dists)
            self.screen.schedule_draw(to_redraw, all = True)
            self.set_Rnx_max_K()


    def open_new_full_dataset(self, dataset_name):
        self.screen.make_dataset_unclickable(dataset_name)
        X, Y, is_labeled, is_classification, colors, is_dists = self.data_loader.get(dataset_name)
        if not is_labeled:
            colors = np.ones((X.shape[0], 3))*np.array([50, 0, 250])
        else:
            if is_classification:
                if colors is None:
                    Y_colors = random_colors(len(np.unique(Y)))
                else:
                    Y_colors = colors.copy()
                colors = np.ones((X.shape[0], 3))
                for idx in range(X.shape[0]):
                    colors[idx] = Y_colors[Y[idx]]
            else:
                colors = np.ones((X.shape[0], 3))
                yellow = np.array([250, 30, 0])
                blue   = np.array([0, 220, 250])
                Y_min, Y_max = np.min(Y), np.max(Y)
                span = Y_max - Y_min + 1e-9
                for idx in range(X.shape[0]):
                    if self.regr_color_using_span:
                        score  = (Y[idx] - Y_min) / span
                    else:
                        score  = max(0., min(1., (Y[idx] - Y_min) / (5*np.std(Y) + 1e-9 ) ))
                    colors[idx] = yellow * score + blue * (1-score)
        self.new_dataset(dataset_name, X, Y, is_labeled, is_classification, colors, is_dists)

    # makes a new dataset class, adds it to the open dataset, and update main_screen
    def new_dataset(self, dataset_name, X, Y, is_labeled, is_classification, colors, is_dists):
        if dataset_name in self.open_datasets and not self.current_dataset == dataset_name:
            self.change_dataset(dataset_name)
        else:
            if is_labeled and is_classification:
                Y = self.sanitize_Y(Y, colors)
            dataset = Dataset(dataset_name, X, Y, is_labeled, is_classification, is_dists) # ici
            self.open_datasets[dataset_name]             = dataset
            self.open_datasets_Ycolors[dataset_name]     = colors
            self.open_datasets_Projections[dataset_name] = []
            self.open_datasets_models[dataset_name] = []
            self.screen.new_tab(dataset_name)
            self.change_dataset(dataset_name)

    def sanitize_Y(self, Y, colors):
        uniques = np.sort(np.unique(Y, return_counts = False))
        mapping = np.arange(np.max(uniques)+1)
        for v in range(uniques.shape[0]):
            mapping[uniques[v]] = v
        new_Y      = np.zeros_like(Y)
        for pt in range(Y.shape[0]):
            new_Y[pt]      = mapping[Y[pt]]
        return new_Y


    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        if pressed_keys[0] == 'o':
            self.open_new_subset_dataset(to_redraw)
        return False
