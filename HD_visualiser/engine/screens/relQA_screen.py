from engine.gui.container import Container
from engine.gui.scatterplot import Scatterplot, Heatmap
from engine.gui.window import Window
from engine.gui.selector import  Button, Mutex_with_title, String_selector, Number_selector, Mutex_choice, Scrollable_bundle
from engine.gui.listener import Listener
from engine.gui.graph import Thumbnail, Comparable_QA_graph, Comparable_Negpos_QA_graph, Comparable_Shepard_diagram, rel_localQA_scatterplot
from engine.gui.event_ids import *
from utils import random_colors
import numpy as np
from utils import luminosity_change

class relQA_screen():
    def __init__(self, theme, window, manager):
        self.manager = manager
        self.eight_colors = None
        self.views = ["distances", "neighbours", "labels"]
        self.color = theme["color"]
        self.color_dark = luminosity_change(theme["color"], -300)
        self.color_left  = np.array([150, 130, 250])
        self.color_right = np.array([250, 150, 110])
        self.selected_view  = "distances"

        self.left_bar     = Container((0,0), (0.15, 1), "left bar rel", window, manager.uid_generator, color=self.color_left, background_color=theme["background"], filled=True)
        self.left_bar.add_line([(1.,0),  (1.,1.), 1, self.left_bar.color])
        self.left_bar.add_rect([(0, 0),  (1.,1.), 10, self.color_left])
        self.right_bar     = Container((0.85,0), (0.15, 1), "right bar rel", window, manager.uid_generator, color=self.color_right, background_color=theme["background"], filled=True)
        self.right_bar.add_line([(1.,0),  (1.,1.), 1, self.right_bar.color])
        self.right_bar.add_rect([(0, 0),  (1.,1.), 10, self.color_right])

        self.top_bar      = Container((0.155,0), (0.695, 0.1), "top bar", window, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.top_bar.add_line([(0.,1),  (0.995,1.), 1, self.top_bar.color])
        self.top_bar.add_leaf(Mutex_choice((0,0.01), (0.6,0.99), "view choice", self.top_bar, uid_generator=manager.uid_generator, nb_col=3, labels = self.views, on_value_change_listener=Listener(ABSQA_VIEW_CHANGE, [self.manager]), ignore_with_ctrl=True))
        self.main_content = Container((0.155,0.105), (0.695, 0.895), "content", window, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)

        window.add_container(self.left_bar)
        window.add_container(self.right_bar)
        window.add_container(self.top_bar)
        window.add_container(self.main_content)


        self.dist_view  = Container((0.,0.), (1, 1), "distance", self.main_content, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.label_view = Container((0.,0.), (1, 1), "label", self.main_content, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.neigh_view = Container((0.,0.), (1, 1), "neighbours", self.main_content, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=True)
        self.main_content.add_container(self.dist_view)
        self.main_content.add_container(self.label_view)
        self.main_content.add_container(self.neigh_view)

        # distance view
        self.dist_top_right    = rel_localQA_scatterplot((0.62, 0.05), (0.38, 0.45), "top right", self.dist_view, X_LD=None, colors_vrand=None, colors_vself=None, mode="vs. random", uid_generator=manager.uid_generator, color=theme["color"])
        self.dist_bottom_right = rel_localQA_scatterplot((0.62, 0.52), (0.38, 0.45), "bottom right", self.dist_view, X_LD=None, colors_vrand=None, colors_vself=None, mode="vs. random", uid_generator=manager.uid_generator, color=theme["color"])
        self.shepard = Comparable_Shepard_diagram((0.01, 0.02), (0.59*(self.dist_view.dim[1]/self.dist_view.dim[0]), 0.59), 'shepard diagram', self.dist_view, color=theme["color"], uid_generator=manager.uid_generator, color_L=self.color_left, color_R=self.color_right)
        self.dist_view.add_leaf(self.dist_top_right)
        self.dist_view.add_leaf(self.dist_bottom_right)
        self.dist_view.add_leaf(self.shepard)

        # labels view
        self.label_top_right    = rel_localQA_scatterplot((0.55, 0.05), (0.38, 0.45), "top right", self.label_view, X_LD=None, colors_vrand=None, colors_vself=None, mode="vs. random", uid_generator=manager.uid_generator, color=theme["color"])
        self.label_bottom_right = rel_localQA_scatterplot((0.55, 0.55), (0.38, 0.45), "bottom right", self.label_view, X_LD=None, colors_vrand=None, colors_vself=None, mode="vs. random", uid_generator=manager.uid_generator, color=theme["color"])
        self.label_bottom_left  = rel_localQA_scatterplot((0.02, 0.55), (0.38, 0.45), "bottom right", self.label_view, X_LD=None, colors_vrand=None, colors_vself=None, mode="vs. random", uid_generator=manager.uid_generator, color=theme["color"])
        self.Knn_gain_plot      = Comparable_Negpos_QA_graph((0.02, 0.02), (0.5, 0.5), 'knn gain', self.label_view, manager.uid_generator)
        self.label_view.add_leaf(self.label_top_right)
        self.label_view.add_leaf(self.label_bottom_right)
        self.label_view.add_leaf(self.label_bottom_left)
        self.label_view.add_leaf(self.Knn_gain_plot)

        # neighbours view
        self.neigh_top_right    = rel_localQA_scatterplot((0.55, 0.05), (0.38, 0.45), "top right", self.neigh_view, X_LD=None, colors_vrand=None, colors_vself=None, mode="vs. random", uid_generator=manager.uid_generator, color=theme["color"])
        self.neigh_bottom_right = rel_localQA_scatterplot((0.55, 0.55), (0.38, 0.45), "bottom right", self.neigh_view, X_LD=None, colors_vrand=None, colors_vself=None, mode="vs. random", uid_generator=manager.uid_generator, color=theme["color"])
        self.neigh_bottom_left  = rel_localQA_scatterplot((0.02, 0.55), (0.38, 0.45), "bottom right", self.neigh_view, X_LD=None, colors_vrand=None, colors_vself=None, mode="vs. random", uid_generator=manager.uid_generator, color=theme["color"])
        self.Rnx_plot           = Comparable_QA_graph((0.02, 0.02), (0.5, 0.5), 'knn gain', self.neigh_view, manager.uid_generator)
        self.neigh_view.add_leaf(self.neigh_top_right)
        self.neigh_view.add_leaf(self.neigh_bottom_right)
        self.neigh_view.add_leaf(self.neigh_bottom_left)
        self.neigh_view.add_leaf(self.Rnx_plot)


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

    def update_local_distQA(self):
        selected_embedding_L = self.manager.selected_embedding_L
        selected_embedding_R = self.manager.selected_embedding_R
        if selected_embedding_L is None or not selected_embedding_L.local_distQA.ready or  selected_embedding_R is None or not selected_embedding_R.local_distQA.ready:
            self.dist_top_right.set_points(None)
            self.dist_bottom_right.set_points(None)
        else:
            self.dist_top_right.set_points(selected_embedding_L.X_LD)
            self.dist_top_right.color_changes(selected_embedding_L.local_distQA.top_right_scores_vs_rand,selected_embedding_R.local_distQA.top_right_scores_vs_rand)
            self.dist_top_right.set_description(selected_embedding_L.local_distQA.top_right_description)
            self.dist_top_right.set_title(selected_embedding_L.local_distQA.top_right_title)

            self.dist_bottom_right.set_points(selected_embedding_L.X_LD)
            self.dist_bottom_right.color_changes(selected_embedding_L.local_distQA.bottom_right_scores_vs_rand, selected_embedding_R.local_distQA.bottom_right_scores_vs_rand)
            self.dist_bottom_right.set_description(selected_embedding_L.local_distQA.bottom_right_description)
            self.dist_bottom_right.set_title(selected_embedding_L.local_distQA.bottom_right_title)
        if self.selected_view == "distances":
            self.manager.ask_redraw(self.main_content)


    def update_local_labelQA(self):
        selected_embedding_L = self.manager.selected_embedding_L
        selected_embedding_R = self.manager.selected_embedding_R
        if selected_embedding_L is None or not selected_embedding_L.local_labelQA.ready or  selected_embedding_R is None or not selected_embedding_R.local_labelQA.ready:
            self.label_top_right.set_points(None)
            self.label_bottom_right.set_points(None)
            self.label_bottom_left.set_points(None)
        else:
            self.label_top_right.set_points(selected_embedding_L.X_LD)
            self.label_top_right.color_changes(selected_embedding_L.local_labelQA.top_right_scores_vs_rand,selected_embedding_R.local_labelQA.top_right_scores_vs_rand)
            self.label_top_right.set_description(selected_embedding_L.local_labelQA.top_right_description)
            self.label_top_right.set_title(selected_embedding_L.local_labelQA.top_right_title)

            self.label_bottom_right.set_points(selected_embedding_L.X_LD)
            self.label_bottom_right.color_changes(selected_embedding_L.local_labelQA.bottom_right_scores_vs_rand,selected_embedding_R.local_labelQA.bottom_right_scores_vs_rand)
            self.label_bottom_right.set_description(selected_embedding_L.local_labelQA.bottom_right_description)
            self.label_bottom_right.set_title(selected_embedding_L.local_labelQA.bottom_right_title)

            self.label_bottom_left.set_points(selected_embedding_L.X_LD)
            self.label_bottom_left.color_changes(selected_embedding_L.local_labelQA.bottom_left_scores_vs_rand,selected_embedding_R.local_labelQA.bottom_left_scores_vs_rand)
            self.label_bottom_left.set_description(selected_embedding_L.local_labelQA.bottom_left_description)
            self.label_bottom_left.set_title(selected_embedding_L.local_labelQA.bottom_left_title)
        if self.selected_view == "labels":
            self.manager.ask_redraw(self.main_content)



    def update_local_neighQA(self):
        selected_embedding_L = self.manager.selected_embedding_L
        selected_embedding_R = self.manager.selected_embedding_R
        if selected_embedding_L is None or not selected_embedding_L.local_neighQA.ready or  selected_embedding_R is None or not selected_embedding_R.local_neighQA.ready:
            self.neigh_top_right.set_points(None)
            self.neigh_bottom_right.set_points(None)
            self.neigh_bottom_left.set_points(None)
        else:
            self.neigh_top_right.set_points(selected_embedding_L.X_LD)
            self.neigh_top_right.color_changes(selected_embedding_L.local_neighQA.top_right_scores_vs_rand,selected_embedding_R.local_neighQA.top_right_scores_vs_rand)
            self.neigh_top_right.set_description(selected_embedding_L.local_neighQA.top_right_description)
            self.neigh_top_right.set_title(selected_embedding_L.local_neighQA.top_right_title)

            self.neigh_bottom_right.set_points(selected_embedding_L.X_LD)
            self.neigh_bottom_right.color_changes(selected_embedding_L.local_neighQA.bottom_right_scores_vs_rand,selected_embedding_R.local_neighQA.bottom_right_scores_vs_rand)
            self.neigh_bottom_right.set_description(selected_embedding_L.local_neighQA.bottom_right_description)
            self.neigh_bottom_right.set_title(selected_embedding_L.local_neighQA.bottom_right_title)

            self.neigh_bottom_left.set_points(selected_embedding_L.X_LD)
            self.neigh_bottom_left.color_changes(selected_embedding_L.local_neighQA.bottom_left_scores_vs_rand,selected_embedding_R.local_neighQA.bottom_left_scores_vs_rand)
            self.neigh_bottom_left.set_description(selected_embedding_L.local_neighQA.bottom_left_description)
            self.neigh_bottom_left.set_title(selected_embedding_L.local_neighQA.bottom_left_title)
        if self.selected_view == "neighbours":
            self.manager.ask_redraw(self.main_content)

    def update_KNNgain_curve(self, proj_name, KNNgain, KNNgain_AUC, change_colors):
        selected_embedding_L = self.manager.selected_embedding_L
        selected_embedding_R = self.manager.selected_embedding_R
        self.Knn_gain_plot.set_points(KNNgain, KNNgain_AUC, proj_name, self.color_dark)
        if change_colors:
            self.Knn_gain_plot.color_L_and_R(selected_embedding_L, selected_embedding_R, self.color_left, self.color_right, self.color_dark)

    def update_Rnx_curve(self, proj_name, Rnx, Rnx_AUC, change_colors):
        selected_embedding_L = self.manager.selected_embedding_L
        selected_embedding_R = self.manager.selected_embedding_R
        self.Rnx_plot.set_points(Rnx, Rnx_AUC, proj_name, self.color_dark)
        if change_colors:
            self.Rnx_plot.color_L_and_R(selected_embedding_L, selected_embedding_R, self.color_left, self.color_right, self.color_dark)

    def update_shepard(self, dataset_D, dist_idxes1, dist_idxes2):
        selected_embedding_L = self.manager.selected_embedding_L
        selected_embedding_R = self.manager.selected_embedding_R
        D_ld_L = None
        D_ld_R = None
        if selected_embedding_L is not None and selected_embedding_L.neighbours_ready:
            D_ld_L = selected_embedding_L.D[dist_idxes1, dist_idxes2]
        if selected_embedding_R is not None and selected_embedding_R.neighbours_ready:
            D_ld_R = selected_embedding_R.D[dist_idxes1, dist_idxes2]

        D_hd = dataset_D[dist_idxes1,   dist_idxes2]
        self.shepard.receive_distances(D_ld_L, D_ld_R, D_hd)
        if self.selected_view == "distances":
            self.manager.ask_redraw(self.main_content)


    def update_Rnx(self, proj_name, Rnx, Rnx_AUC, change_colors):
        self.update_Rnx_curve(proj_name, Rnx, Rnx_AUC, change_colors)
        self.manager.ask_redraw(self.main_content)

    def update_KNNgain(self, proj_name, KNNgain, KNNgain_AUC, change_colors):
        self.update_KNNgain_curve(proj_name, KNNgain, KNNgain_AUC, change_colors)
        self.manager.ask_redraw(self.main_content)

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
            thumbnail = self.right_bar.leaves[i]
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







    def build_dist_view(self, dataset, do_shepard):
        # shepard diagram
        if do_shepard and dataset is not None:
            self.update_shepard(dataset.D, dataset.shepard_distances1, dataset.shepard_distances2)
        # local QA
        self.update_local_distQA()


    def build_thumbnails(self, embeddings):
        print("building thumbnails")
        selected_L_name = self.manager.selected_embedding_L_name
        selected_R_name = self.manager.selected_embedding_R_name
        N_projections = len(embeddings)
        for i in range(8):
            thumbnail_L = self.left_bar.leaves[i]
            thumbnail_R = self.right_bar.leaves[i]
            if i+1 <= N_projections and embeddings[i].done:
                thumbnail_points = embeddings[i].X_LD[:min(1000,embeddings[i].X_LD.shape[0]-1)]
                title, Dcorr, RnxAUC, KNNgain = embeddings[i].proj_name, "", "", ""
                if embeddings[i].generalQA.Rnx_ready:
                    RnxAUC = str(embeddings[i].generalQA.Rnx_AUC)
                if embeddings[i].generalQA.KNNgain_ready:
                    KNNgain = str(embeddings[i].generalQA.KNNgain_AUC)
                if embeddings[i].generalQA.Dcorr_ready:
                    Dcorr = str(embeddings[i].generalQA.Dcorr)
                thumbnail_L.set_points(thumbnail_points)
                thumbnail_R.set_points(thumbnail_points)
                thumbnail_L.set_text(title, Dcorr, RnxAUC, KNNgain)
                thumbnail_R.set_text(title, Dcorr, RnxAUC, KNNgain)
                thumbnail_L.name = embeddings[i].proj_name
                thumbnail_R.name = embeddings[i].proj_name
            else:
                thumbnail_L.set_points(None)
                thumbnail_R.set_points(None)
                thumbnail_L.set_text("", "", "", "")
                thumbnail_R.set_text("", "", "", "")
                thumbnail_L.name = "thumbnail "+str(i+1)
                thumbnail_R.name = "thumbnail "+str(i+1)
            if thumbnail_L.name == selected_L_name:
                thumbnail_L.selected = True
            if thumbnail_R.name == selected_R_name:
                thumbnail_R.selected = True


    def build_all(self, embeddings, to_redraw=None, dataset=None):
        change_colors = False
        N_embeddings = len(embeddings)
        self.build_thumbnails(embeddings)
        for idx, embedding in enumerate(embeddings):
            if idx == N_embeddings-1:
                change_colors = True
            if embedding is not None and not embedding.deleted and embedding.local_labelQA is not None and not embedding.local_labelQA.deleted and embedding.local_labelQA.ready:
                self.update_KNNgain_curve(embedding.proj_name, embedding.generalQA.KNNgain, embedding.generalQA.KNNgain_AUC, change_colors=change_colors)
            if embedding is not None and not embedding.deleted and embedding.local_neighQA is not None and not embedding.local_neighQA.deleted and embedding.local_neighQA.ready:
                self.update_Rnx_curve(embedding.proj_name, embedding.generalQA.Rnx, embedding.generalQA.Rnx_AUC, change_colors=change_colors)

        self.build_dist_view(dataset, do_shepard = True)
        self.update_local_labelQA()
        self.update_local_neighQA()

        if to_redraw is not None:
            self.left_bar.schedule_draw(to_redraw)
            self.right_bar.schedule_draw(to_redraw)
            self.top_bar.schedule_draw(to_redraw)
            self.main_content.schedule_draw(to_redraw)


    def projection_selected(self, proj_name, embedding, left, to_redraw):
        if left:
            for thumbnail in self.left_bar.leaves:
                if thumbnail.name == proj_name:
                    thumbnail.selected = True
                else:
                    thumbnail.selected = False
        else:
            for thumbnail in self.right_bar.leaves:
                if thumbnail.name == proj_name:
                    thumbnail.selected = True
                else:
                    thumbnail.selected = False
        if self.manager.main_manager.current_dataset != "new" and self.manager.main_manager.current_dataset is not None:
            self.build_all(self.manager.main_manager.open_datasets_Projections[self.manager.main_manager.current_dataset], to_redraw, self.manager.get_dataset())

    def generate_thumbnails(self):
        for i in range(8):
            listener_L = Listener(THUMBNAIL_CLICKED_L, [self.manager])
            self.left_bar.add_leaf(Thumbnail((0.05, 0.05 + i*0.9/8), (0.9,  0.9/8), "thumbnail "+str(i+1), self.left_bar, None, None, None, None, listener_L, self.manager.uid_generator, color=self.eight_colors[i], show_scores=False))
            listener_R = Listener(THUMBNAIL_CLICKED_R, [self.manager])
            self.right_bar.add_leaf(Thumbnail((0.05, 0.05 +  i*0.9/8), (0.9, 0.9/8), "thumbnail "+str(i+1), self.right_bar, None, None, None, None, listener_R, self.manager.uid_generator, color=self.eight_colors[i], show_scores=False))
