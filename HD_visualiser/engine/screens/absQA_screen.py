from engine.gui.container import Container
from engine.gui.scatterplot import Scatterplot, Heatmap
from engine.gui.element import Element, Text
from engine.gui.window import Window
from engine.gui.selector import  Button, Mutex_with_title, String_selector, Number_selector, Mutex_choice, Scrollable_bundle
from engine.gui.listener import Listener
from engine.gui.graph import QA_graph, Thumbnail, abs_localQA_scatterplot, Shepard_diagram, Negpos_QA_graph, compute_colors, shepard_dist_to_px
from engine.gui.event_ids import *
from utils import random_colors
import numpy as np
from utils import luminosity_change

def precompile():
    print("precompiling some JIT things...")
    compute_colors(np.array([0.2, 0.3 , 0.4]), np.array([0.2, 0.3 , 0.4]), 0.4)
    shepard_dist_to_px(np.random.uniform(size = 200), np.random.uniform(size = 200), 10)
    print("done")

class absQA_screen():
    def __init__(self, theme, window, manager):
        precompile()
        self.manager = manager
        self.eight_colors = None
        self.views = ["distances", "neighbours", "labels"]
        self.color = theme["color"]
        self.color_dark = luminosity_change(theme["color"], -300)

        self.selected_view  = "distances"
        self.threshold      = 0.68
        self.versus         = "vs. random"

        self.left_bar     = Container((0,0), (0.15, 1), "left bar", window, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.left_bar.add_line([(1.,0),  (1.,1.), 1, self.left_bar.color])
        self.top_bar      = Container((0.155,0), (0.845, 0.1), "top bar", window, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.top_bar.add_line([(0.,1),  (0.995,1.), 1, self.top_bar.color])
        self.top_bar.add_leaf(Mutex_choice((0,0.01), (0.4,0.99), "view choice", self.top_bar, uid_generator=manager.uid_generator, nb_col=3, labels = self.views, on_value_change_listener=Listener(ABSQA_VIEW_CHANGE, [self.manager]), ignore_with_ctrl=True))
        self.top_bar.add_leaf(Mutex_choice((0.45,0.01), (0.3,0.99), "versus", self.top_bar, uid_generator=manager.uid_generator, nb_col=2, labels = ["vs. random", "vs. self"], on_value_change_listener=Listener(ABSQA_VERSUS_CHANGE, [self.manager]), ignore_with_ctrl=True))
        self.top_bar.add_leaf(Number_selector((0.75,0.01), (0.25,0.99), "threshold", self.top_bar, manager.uid_generator, "float", 0.0, 1., 0.025, 0.68, 0.68, on_value_change_listener=Listener(ABSQA_THRESHOLD_CHANGE, [self.manager])))
        self.main_content = Container((0.155,0.105), (0.845, 0.895), "content", window, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)

        window.add_container(self.left_bar)
        window.add_container(self.top_bar)
        window.add_container(self.main_content)


        self.dist_view  = Container((0.,0.), (1, 1), "distance", self.main_content, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.label_view = Container((0.,0.), (1, 1), "label", self.main_content, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.neigh_view = Container((0.,0.), (1, 1), "neighbours", self.main_content, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.main_content.add_container(self.dist_view)
        self.main_content.add_container(self.label_view)
        self.main_content.add_container(self.neigh_view)

        # distance view
        self.dist_top_right    = abs_localQA_scatterplot((0.62, 0.05), (0.38, 0.45), "top right", self.dist_view, X_LD=None, colors_vrand=None, colors_vself=None, mode=self.versus, uid_generator=manager.uid_generator, color=theme["color"])
        self.dist_bottom_right = abs_localQA_scatterplot((0.62, 0.52), (0.38, 0.45), "bottom right", self.dist_view, X_LD=None, colors_vrand=None, colors_vself=None, mode=self.versus, uid_generator=manager.uid_generator, color=theme["color"])
        self.shepard = Shepard_diagram((0.02, 0.02), (0.7*(self.dist_view.dim[1]/self.dist_view.dim[0]), 0.7), 'shepard diagram', self.dist_view, color=theme["color"], uid_generator=manager.uid_generator)
        self.dist_view.add_leaf(self.dist_top_right)
        self.dist_view.add_leaf(self.dist_bottom_right)
        self.dist_view.add_leaf(self.shepard)

        # labels view
        self.label_top_right    = abs_localQA_scatterplot((0.52, 0.05), (0.38, 0.45), "top right", self.label_view, X_LD=None, colors_vrand=None, colors_vself=None, mode=self.versus, uid_generator=manager.uid_generator, color=theme["color"])
        self.label_bottom_right = abs_localQA_scatterplot((0.52, 0.52), (0.38, 0.45), "bottom right", self.label_view, X_LD=None, colors_vrand=None, colors_vself=None, mode=self.versus, uid_generator=manager.uid_generator, color=theme["color"])
        self.label_bottom_left  = abs_localQA_scatterplot((0.02, 0.52), (0.38, 0.45), "bottom right", self.label_view, X_LD=None, colors_vrand=None, colors_vself=None, mode=self.versus, uid_generator=manager.uid_generator, color=theme["color"])
        self.Knn_gain_plot      = Negpos_QA_graph((0.02, 0.02), (0.5, 0.5), 'knn gain', self.label_view, manager.uid_generator)
        self.label_view.add_leaf(self.label_top_right)
        self.label_view.add_leaf(self.label_bottom_right)
        self.label_view.add_leaf(self.label_bottom_left)
        self.label_view.add_leaf(self.Knn_gain_plot)

        # neighbours view
        self.neigh_top_right    = abs_localQA_scatterplot((0.52, 0.05), (0.38, 0.45), "top right", self.neigh_view, X_LD=None, colors_vrand=None, colors_vself=None, mode=self.versus, uid_generator=manager.uid_generator, color=theme["color"])
        self.neigh_bottom_right = abs_localQA_scatterplot((0.52, 0.52), (0.38, 0.45), "bottom right", self.neigh_view, X_LD=None, colors_vrand=None, colors_vself=None, mode=self.versus, uid_generator=manager.uid_generator, color=theme["color"])
        self.neigh_bottom_left  = abs_localQA_scatterplot((0.02, 0.52), (0.38, 0.45), "bottom right", self.neigh_view, X_LD=None, colors_vrand=None, colors_vself=None, mode=self.versus, uid_generator=manager.uid_generator, color=theme["color"])
        self.Rnx_plot           = QA_graph((0.02, 0.02), (0.5, 0.5), 'knn gain', self.neigh_view, manager.uid_generator, auto_set_max_k=True)
        self.neigh_view.add_leaf(self.neigh_top_right)
        self.neigh_view.add_leaf(self.neigh_bottom_right)
        self.neigh_view.add_leaf(self.neigh_bottom_left)
        self.neigh_view.add_leaf(self.Rnx_plot)




    def versus_change(self, versus, embedding, to_redraw, dataset):
        self.versus = versus
        self.build_dist_view(embedding, dataset, do_shepard = False)
        self.build_label_view(embedding, do_KNNgain = False)
        self.build_neigh_view(embedding, do_Rnx = False)
        if to_redraw is not None:
            self.main_content.schedule_draw(to_redraw)

    def threshold_change(self, threshold, embedding, to_redraw, dataset):
        self.threshold = threshold
        self.build_dist_view(embedding, dataset, do_shepard = False)
        self.build_label_view(embedding, do_KNNgain = False)
        self.build_neigh_view(embedding, do_Rnx = False)
        if to_redraw is not None:
            self.main_content.schedule_draw(to_redraw)

    def view_change(self, view_name, to_redraw):
        self.selected_view = view_name
        if view_name == "distances":
            self.dist_view.enable()
            self.label_view.disable()
            self.neigh_view.disable()
        elif view_name == "neighbours":
            self.dist_view.disable()
            self.label_view.disable()
            self.neigh_view.enable()
        elif view_name == "labels":
            self.dist_view.disable()
            self.label_view.enable()
            self.neigh_view.disable()
        if to_redraw is not None:
            self.left_bar.schedule_draw(to_redraw)
            self.main_content.schedule_draw(to_redraw)

    def update_local_distQA(self, embedding):
        self.dist_top_right.mode    = self.versus
        self.dist_bottom_right.mode = self.versus
        if embedding.local_distQA is not None and not embedding.local_distQA.deleted and embedding.local_distQA.ready:
            self.dist_top_right.set_points(embedding.X_LD)
            self.dist_top_right.color_changes(embedding.local_distQA.top_right_scores_vs_rand, embedding.local_distQA.top_right_scores_vs_self, self.threshold)
            self.dist_top_right.set_description(embedding.local_distQA.top_right_description)
            self.dist_top_right.set_overall_score(embedding.local_distQA.top_right_overall_score)
            self.dist_top_right.set_title(embedding.local_distQA.top_right_title)


            self.dist_bottom_right.set_points(embedding.X_LD)
            self.dist_bottom_right.color_changes(embedding.local_distQA.bottom_right_scores_vs_rand, embedding.local_distQA.bottom_right_scores_vs_self, self.threshold)
            self.dist_bottom_right.set_description(embedding.local_distQA.bottom_right_description)
            self.dist_bottom_right.set_overall_score(embedding.local_distQA.bottom_right_overall_score)
            self.dist_bottom_right.set_title(embedding.local_distQA.bottom_right_title)
            if self.selected_view == "distances":
                self.manager.ask_redraw(self.main_content)


    def update_local_labelQA(self, embedding):
        self.label_top_right.mode      =  self.versus
        self.label_bottom_right.mode   =  self.versus
        self.label_bottom_left.mode    =  self.versus
        if embedding.local_labelQA is not None and not embedding.local_labelQA.deleted and embedding.local_labelQA.ready:
            self.label_top_right.set_points(embedding.X_LD)
            self.label_top_right.color_changes(embedding.local_labelQA.top_right_scores_vs_rand, embedding.local_labelQA.top_right_scores_vs_self, self.threshold)
            self.label_top_right.set_description(embedding.local_labelQA.top_right_description)
            self.label_top_right.set_overall_score(embedding.local_labelQA.top_right_overall_score)
            self.label_top_right.set_title(embedding.local_labelQA.top_right_title)

            self.label_bottom_right.set_points(embedding.X_LD)
            self.label_bottom_right.color_changes(embedding.local_labelQA.bottom_right_scores_vs_rand, embedding.local_labelQA.bottom_right_scores_vs_self, self.threshold)
            self.label_bottom_right.set_description(embedding.local_labelQA.bottom_right_description)
            self.label_bottom_right.set_overall_score(embedding.local_labelQA.bottom_right_overall_score)
            self.label_bottom_right.set_title(embedding.local_labelQA.bottom_right_title)

            self.label_bottom_left.set_points(embedding.X_LD)
            self.label_bottom_left.color_changes(embedding.local_labelQA.bottom_left_scores_vs_rand, embedding.local_labelQA.bottom_left_scores_vs_self, self.threshold)
            self.label_bottom_left.set_description(embedding.local_labelQA.bottom_left_description)
            self.label_bottom_left.set_overall_score(embedding.local_labelQA.bottom_left_overall_score)
            self.label_bottom_left.set_title(embedding.local_labelQA.bottom_left_title)
            if self.selected_view == "labels":
                self.manager.ask_redraw(self.main_content)


    def update_local_neighQA(self, embedding):
        self.neigh_top_right.mode      =  self.versus
        self.neigh_bottom_right.mode   =  self.versus
        self.neigh_bottom_left.mode    =  self.versus
        if embedding.local_neighQA is not None and not embedding.local_neighQA.deleted and embedding.local_neighQA.ready:
            self.neigh_top_right.set_points(embedding.X_LD)
            self.neigh_top_right.color_changes(embedding.local_neighQA.top_right_scores_vs_rand, embedding.local_neighQA.top_right_scores_vs_self, self.threshold)
            self.neigh_top_right.set_description(embedding.local_neighQA.top_right_description)
            self.neigh_top_right.set_overall_score(embedding.local_neighQA.top_right_overall_score)
            self.neigh_top_right.set_title(embedding.local_neighQA.top_right_title)

            self.neigh_bottom_right.set_points(embedding.X_LD)
            self.neigh_bottom_right.color_changes(embedding.local_neighQA.bottom_right_scores_vs_rand, embedding.local_neighQA.bottom_right_scores_vs_self, self.threshold)
            self.neigh_bottom_right.set_description(embedding.local_neighQA.bottom_right_description)
            self.neigh_bottom_right.set_overall_score(embedding.local_neighQA.bottom_right_overall_score)
            self.neigh_bottom_right.set_title(embedding.local_neighQA.bottom_right_title)

            self.neigh_bottom_left.set_points(embedding.X_LD)
            self.neigh_bottom_left.color_changes(embedding.local_neighQA.bottom_left_scores_vs_rand, embedding.local_neighQA.bottom_left_scores_vs_self, self.threshold)
            self.neigh_bottom_left.set_description(embedding.local_neighQA.bottom_left_description)
            self.neigh_bottom_left.set_overall_score(embedding.local_neighQA.bottom_left_overall_score)
            self.neigh_bottom_left.set_title(embedding.local_neighQA.bottom_left_title)
            if self.selected_view == "neighbours":
                self.manager.ask_redraw(self.main_content)

    def update_KNNgain_curve(self, proj_name, KNNgain, KNNgain_AUC, is_selected_embedding):
        self.Knn_gain_plot.set_points(KNNgain, KNNgain_AUC, proj_name, self.color_dark)
        if is_selected_embedding:
            self.Knn_gain_plot.change_selected(proj_name, self.color, self.color_dark)

    def update_Rnx_curve(self, proj_name, Rnx, Rnx_AUC, is_selected_embedding):
        self.Rnx_plot.set_points(Rnx, Rnx_AUC, proj_name, self.color_dark)
        if is_selected_embedding:
            self.Rnx_plot.change_selected(proj_name, self.color, self.color_dark)

    def update_shepard(self, embedding_D, dataset_D, dist_idxes1, dist_idxes2):
        D_ld    = embedding_D[dist_idxes1, dist_idxes2]
        D_hd    = dataset_D[dist_idxes1,   dist_idxes2]
        self.shepard.receive_distances(D_ld, D_hd)
        if self.selected_view == "distances":
            self.manager.ask_redraw(self.main_content)

    def update_Rnx(self, proj_name, Rnx, Rnx_AUC, is_selected_embedding):
        for i in range(8):
            thumbnail = self.left_bar.leaves[i]
            if thumbnail.name == proj_name:
                thumbnail.RnxAUC = Rnx_AUC
                thumbnail.RnxAUC_text = Text((0.6, 0.5), 1, (0.5, 0.5), 16, "Rnx "+str(Rnx_AUC), thumbnail, draw_background=True)
        self.update_Rnx_curve(proj_name, Rnx, Rnx_AUC, is_selected_embedding)
        self.manager.ask_redraw(self.main_content)
        self.manager.ask_redraw(self.left_bar)

    def update_KNNgain(self, proj_name, KNNgain, KNNgain_AUC, is_selected_embedding):
        for i in range(8):
            thumbnail = self.left_bar.leaves[i]
            if thumbnail.name == proj_name:
                thumbnail.KNNgain = KNNgain_AUC
                thumbnail.KNNgain_text = Text((0.6, 0.7), 1, (0.5, 0.5), 16, "KNN "+str(KNNgain_AUC), thumbnail, draw_background=True)
        self.update_KNNgain_curve(proj_name, KNNgain, KNNgain_AUC, is_selected_embedding)
        self.manager.ask_redraw(self.main_content)
        self.manager.ask_redraw(self.left_bar)

    def update_Dcorr(self, proj_name, Dcorr):
        for i in range(8):
            thumbnail = self.left_bar.leaves[i]
            if thumbnail.name == proj_name:
                thumbnail.Dcorr = Dcorr
                thumbnail.Dcorr_text = Text((0.6, 0.3), 1, (0.5, 0.5), 16, "Dcorr "+str(Dcorr), thumbnail, draw_background=True)
        self.manager.ask_redraw(self.left_bar)


    def clear_dist_view(self):
        self.dist_top_right.set_points(None)
        self.dist_bottom_right.set_points(None)
        self.shepard.wipe()
        self.Knn_gain_plot.wipe()

    def clear_label_view(self):
        self.label_top_right.set_points(None)
        self.label_bottom_right.set_points(None)
        self.label_bottom_left.set_points(None)
        self.Knn_gain_plot.wipe()

    def clear_neigh_view(self):
        self.neigh_top_right.set_points(None)
        self.neigh_bottom_right.set_points(None)
        self.neigh_bottom_left.set_points(None)
        self.Rnx_plot.wipe()

    def clear_thumbnails(self):
        for i in range(8):
            thumbnail = self.left_bar.leaves[i]
            thumbnail.set_points(None)
            thumbnail.set_text("", "", "", "")
            thumbnail.name = "thumbnail "+str(i+1)
            thumbnail.selected = False

    def clear_all(self, thumbnails_too):
        self.clear_dist_view()
        self.clear_label_view()
        self.clear_neigh_view()
        if thumbnails_too:
            self.clear_thumbnails()









    def build_dist_view(self, embedding, dataset, do_shepard):
        if embedding is None or embedding.deleted or embedding.generalQA is None:
            return
        # shepard diagram
        if do_shepard and dataset is not None and dataset.initialised and embedding.generalQA.Dcorr_ready and embedding.neighbours_ready:
            self.update_shepard(embedding.D, dataset.D, dataset.shepard_distances1, dataset.shepard_distances2)
        # local QA
        self.update_local_distQA(embedding)


    def build_label_view(self, embedding, do_KNNgain):
        if embedding is None or embedding.deleted or embedding.generalQA is None:
            return
        if do_KNNgain and embedding.generalQA.KNNgain_ready and embedding.neighbours_ready:
            self.update_KNNgain(embedding.proj_name, embedding.generalQA.KNNgain, embedding.generalQA.KNNgain_AUC, embedding.proj_name == self.manager.selected_proj_name)
        self.update_local_labelQA(embedding)

    def build_neigh_view(self, embedding, do_Rnx):
        if embedding is None or embedding.deleted or embedding.generalQA is None:
            return
        if do_Rnx and embedding.generalQA.Rnx_ready and embedding.neighbours_ready:
            self.update_Rnx(embedding.proj_name, embedding.generalQA.Rnx, embedding.generalQA.Rnx_AUC, embedding.proj_name == self.manager.selected_proj_name)
        self.update_local_neighQA(embedding)

    def build_thumbnails(self, selected_proj_name, embeddings):
        N_projections = len(embeddings)
        for i in range(8):
            thumbnail = self.left_bar.leaves[i]
            if i+1 <= N_projections and embeddings[i].done:
                thumbnail_points = embeddings[i].X_LD[:min(1000,embeddings[i].X_LD.shape[0]-1)]
                title, Dcorr, RnxAUC, KNNgain = embeddings[i].proj_name, "", "", ""
                if embeddings[i].generalQA.Rnx_ready:
                    RnxAUC = str(embeddings[i].generalQA.Rnx_AUC)
                if embeddings[i].generalQA.KNNgain_ready:
                    KNNgain = str(embeddings[i].generalQA.KNNgain_AUC)
                if embeddings[i].generalQA.Dcorr_ready:
                    Dcorr = str(embeddings[i].generalQA.Dcorr)
                thumbnail.set_points(thumbnail_points)
                thumbnail.set_text(title, Dcorr, RnxAUC, KNNgain)
                thumbnail.name = embeddings[i].proj_name
            else:
                thumbnail.set_points(None)
                thumbnail.set_text("", "", "", "")
                thumbnail.name = "thumbnail "+str(i+1)
            if thumbnail.name == selected_proj_name:
                thumbnail.selected = True
            else:
                thumbnail.selected = False


    def build_all(self, selected_proj_name, embeddings, to_redraw=None, dataset=None):
        self.build_thumbnails(selected_proj_name, embeddings)
        for embedding in embeddings:
            if embedding.proj_name == selected_proj_name:
                self.build_dist_view(embedding, dataset, do_shepard = True)
                self.build_label_view(embedding, do_KNNgain = True)
                self.build_neigh_view(embedding, do_Rnx = True)
            else:
                if embedding is not None and not embedding.deleted and embedding.local_labelQA is not None and not embedding.local_labelQA.deleted and embedding.local_labelQA.ready:
                    self.update_KNNgain_curve(embedding.proj_name, embedding.generalQA.KNNgain, embedding.generalQA.KNNgain_AUC, False)
                if embedding is not None and not embedding.deleted and embedding.local_neighQA is not None and not embedding.local_neighQA.deleted and embedding.local_neighQA.ready:
                    self.update_Rnx_curve(embedding.proj_name, embedding.generalQA.Rnx, embedding.generalQA.Rnx_AUC, False)

        if to_redraw is not None:
            self.left_bar.schedule_draw(to_redraw)
            self.top_bar.schedule_draw(to_redraw)
            self.main_content.schedule_draw(to_redraw)



    def projection_selected(self, proj_name, embedding, to_redraw):
        for thumbnail in self.left_bar.leaves:
            if thumbnail.name == proj_name:
                thumbnail.selected = True
            else:
                thumbnail.selected = False
        if self.manager.main_manager.current_dataset != "new" and self.manager.main_manager.current_dataset is not None:
            self.build_all(proj_name, self.manager.main_manager.open_datasets_Projections[self.manager.main_manager.current_dataset], to_redraw, self.manager.get_dataset())

    def generate_thumbnails(self):
        for i in range(8):
            listener = Listener(THUMBNAIL_CLICKED, [self.manager])
            self.left_bar.add_leaf(Thumbnail((0, i/8), (1, 1/8), "thumbnail "+str(i+1), self.left_bar, None, None, None, None, listener, self.manager.uid_generator, color=self.eight_colors[i]))
