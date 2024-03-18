from engine.screen_managers.manager import Manager
from engine.screens.relQA_screen import relQA_screen
import threading
from engine.gui.event_ids import *


class relQA_manager(Manager):
    def __init__(self, window, theme, uid_generator):
        super(relQA_manager, self).__init__("relQA", initial_state=True)
        self.type_identifier = "relQA_manager"
        self.main_manager  = None
        self.last_dataset_name  = "None"
        self.selected_embedding_L_name = None
        self.selected_embedding_R_name = None
        self.selected_embedding_L = None
        self.selected_embedding_R = None
        self.uid_generator = uid_generator
        self.active  = False
        self.deleted = False
        self.window  = window
        self.window.awaiting_key_press.append(self)
        self.screen  = relQA_screen(theme, window, self)
        self.lock = threading.Lock()

    def closed_projection(self, proj_name):
        if proj_name == self.selected_embedding_L_name:
            self.set_selected(None, None, True)
        if proj_name == self.selected_embedding_R_name:
            self.set_selected(None, None, False)
        self.screen.clear_all(thumbnails_too = True)

    def ask_full_redraw(self):
        self.ask_redraw(self.screen.left_bar)
        self.ask_redraw(self.screen.right_bar)
        self.ask_redraw(self.screen.top_bar)
        self.ask_redraw(self.screen.main_content)

    def attach_main_manager(self, main_manager):
        self.main_manager        = main_manager
        self.screen.eight_colors = main_manager.screen.eight_colors
        self.screen.generate_thumbnails()

    def wake_up(self, prev_manager):
        super(relQA_manager, self).wake_up(prev_manager)
        if self.main_manager.current_dataset != self.last_dataset_name:
            self.screen.clear_all(thumbnails_too = True)
        self.screen.view_change(self.screen.selected_view, None)
        self.rebuild_all(self.main_manager.current_dataset, first_rebuild=True)

    def set_selected(self, embedding, embedding_name, L):
        if L:
            self.selected_embedding_L_name = embedding_name
            self.selected_embedding_L      = embedding
        else:
            self.selected_embedding_R_name = embedding_name
            self.selected_embedding_R      = embedding

    def rebuild_all(self, dataset_name, first_rebuild):
        if dataset_name == "new" or dataset_name is None:
            self.last_dataset = "None"
            self.set_selected(None, None, L = True)
            self.set_selected(None, None, L = False)
            self.screen.clear_all(thumbnails_too = True)
        else:
            self.last_dataset  = dataset_name
            embeddings = self.main_manager.open_datasets_Projections[dataset_name]
            if first_rebuild:
                self.set_selected(self.main_manager.get_selected_embedding(), self.main_manager.get_selected_embedding_name(), L = True)
            if self.selected_embedding_L is None or self.selected_embedding_R is None:
                self.screen.build_thumbnails(embeddings)
                self.screen.clear_all(thumbnails_too = False)
            else:
                self.screen.build_all(embeddings,  dataset=self.get_dataset())
        if not first_rebuild:
            self.ask_full_redraw()

    def get_embedding(self, proj_name, L):
        if proj_name is None:
            if L:
                return self.selected_embedding_L
            else:
                return self.selected_embedding_R
        if self.main_manager.current_dataset != "new":
            for embedding in self.main_manager.open_datasets_Projections[self.main_manager.current_dataset]:
                if embedding.proj_name == proj_name:
                    return embedding
        return None

    def get_dataset(self):
        if self.main_manager.current_dataset != "new":
            return self.main_manager.open_datasets[self.main_manager.current_dataset]
        return None

    def receive_local_distQA(self, dataset_name, proj_name, local_distQA, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        if proj_name is not None and proj_name in [self.selected_embedding_L_name, self.selected_embedding_R_name]:
            self.screen.update_local_distQA()

    def receive_local_neighQA(self, dataset_name, proj_name, local_distQA, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        if proj_name is not None and proj_name in [self.selected_embedding_L_name, self.selected_embedding_R_name]:
            self.screen.update_local_neighQA()

    def receive_local_labelQA(self, dataset_name, proj_name, local_distQA, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        if proj_name is not None and proj_name in [self.selected_embedding_L_name, self.selected_embedding_R_name]:
            self.screen.update_local_labelQA()

    def receive_shepard_data(self, dataset_name, proj_name, embedding_D, dataset_D, dist_idxes1, dist_idxes2, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        if proj_name is not None and proj_name in [self.selected_embedding_L_name, self.selected_embedding_R_name]:
            self.screen.update_shepard(dataset_D, dist_idxes1, dist_idxes2)

    def receive_Rnx_data(self, dataset_name, proj_name, Rnx, Rnx_AUC, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        self.screen.update_Rnx(proj_name, Rnx, Rnx_AUC, True)

    def receive_KNNgain_data(self, dataset_name, proj_name, KNNgain, KNNgain_AUC, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        self.screen.update_KNNgain(proj_name, KNNgain, KNNgain_AUC, True)

    def receive_Dcorr_data(self, dataset_name, proj_name, Dcorr, to_redraw):
        pass

    def get_notified(self, event_class, id, value, to_redraw = []):
        with self.lock:
            if event_class == THUMBNAIL_CLICKED_L:
                self.thumbnail_clicked(value, to_redraw, True)
            elif event_class == THUMBNAIL_CLICKED_R:
                self.thumbnail_clicked(value, to_redraw, False)
            elif event_class == ABSQA_VIEW_CHANGE:
                self.screen.view_change(value, to_redraw)


    def thumbnail_clicked(self, proj_name, to_redraw, L):
        if L:
            if self.selected_embedding_L_name != proj_name:
                embedding = self.get_embedding(proj_name, True)
                self.set_selected(embedding, proj_name, L=True)
                if embedding is not None:
                    self.screen.projection_selected(self.selected_embedding_L_name, embedding, L, to_redraw)
        else:
            if self.selected_embedding_R_name != proj_name:
                embedding = self.get_embedding(proj_name, False)
                self.set_selected(embedding, proj_name, L=False)
                if embedding is not None:
                    self.screen.projection_selected(self.selected_embedding_R_name, embedding, L, to_redraw)

    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        if pressed_keys[0] == 'h':
            print("print scatterplot description")
        return False
