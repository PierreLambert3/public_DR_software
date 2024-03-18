from engine.screen_managers.manager import Manager
from engine.screens.absQA_screen import absQA_screen
import threading
from engine.gui.event_ids import *

class absQA_manager(Manager):
    def __init__(self, window, theme, uid_generator):
        super(absQA_manager, self).__init__("absQA", initial_state=True)
        self.type_identifier = "absQA_manager"
        self.main_manager  = None
        self.last_dataset_name  = "None"
        self.selected_proj_name = None
        self.selected_embedding = None
        self.uid_generator = uid_generator
        self.active  = False
        self.deleted = False
        self.window  = window
        self.window.awaiting_key_press.append(self)
        self.screen  = absQA_screen(theme, window, self)
        self.lock = threading.Lock()

    def ask_full_redraw(self):
        self.ask_redraw(self.screen.left_bar)
        self.ask_redraw(self.screen.main_content)
        self.ask_redraw(self.screen.top_bar)

    def attach_main_manager(self, main_manager):
        self.main_manager        = main_manager
        self.screen.eight_colors = main_manager.screen.eight_colors
        self.screen.generate_thumbnails()

    def wake_up(self, prev_manager):
        super(absQA_manager, self).wake_up(prev_manager)
        if self.main_manager.current_dataset != self.last_dataset_name:
            self.screen.clear_all(thumbnails_too = True)
        self.screen.view_change(self.screen.selected_view, None)
        self.rebuild_all(self.main_manager.current_dataset, first_rebuild=True)

    def closed_projection(self, proj_name):
        if proj_name == self.selected_proj_name:
            self.selected_proj_name = None
            self.selected_embedding = None
        self.screen.clear_all(thumbnails_too = True)

    def rebuild_all(self, dataset_name, first_rebuild):
        if dataset_name == "new" or dataset_name is None:
            self.last_dataset = "None"
            self.selected_proj_name = None
            self.selected_embedding = None
            self.screen.clear_all(thumbnails_too = True)
        else:
            self.last_dataset  = dataset_name
            embeddings = self.main_manager.open_datasets_Projections[dataset_name]
            if first_rebuild:
                self.selected_proj_name = self.main_manager.get_selected_embedding_name()
                self.selected_embedding = self.main_manager.get_selected_embedding()
            if self.selected_embedding is None:
                self.screen.build_thumbnails(None, embeddings)
                self.screen.clear_all(thumbnails_too = False)
            else:
                self.screen.build_all(self.selected_proj_name, embeddings,  dataset=self.get_dataset())
        if not first_rebuild:
            self.ask_full_redraw()

    def get_embedding(self, proj_name):
        if proj_name is None:
            return self.selected_embedding
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
        if proj_name is not None and proj_name == self.selected_proj_name and self.selected_embedding is not None:
            self.screen.update_local_distQA(self.selected_embedding)

    def receive_local_neighQA(self, dataset_name, proj_name, local_distQA, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        if proj_name is not None and proj_name == self.selected_proj_name and self.selected_embedding is not None:
            self.screen.update_local_neighQA(self.selected_embedding)

    def receive_local_labelQA(self, dataset_name, proj_name, local_distQA, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        if proj_name is not None and proj_name == self.selected_proj_name and self.selected_embedding is not None:
            self.screen.update_local_labelQA(self.selected_embedding)

    def receive_shepard_data(self, dataset_name, proj_name, embedding_D, dataset_D, dist_idxes1, dist_idxes2, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        if proj_name is not None and proj_name == self.selected_proj_name and self.selected_embedding is not None:
            self.screen.update_shepard(embedding_D, dataset_D, dist_idxes1, dist_idxes2)

    def receive_Rnx_data(self, dataset_name, proj_name, Rnx, Rnx_AUC, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        self.screen.update_Rnx(proj_name, Rnx, Rnx_AUC, proj_name == self.selected_proj_name)

    def receive_KNNgain_data(self, dataset_name, proj_name, KNNgain, KNNgain_AUC, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        self.screen.update_KNNgain(proj_name, KNNgain, KNNgain_AUC, proj_name == self.selected_proj_name)

    def receive_Dcorr_data(self, dataset_name, proj_name, Dcorr, to_redraw):
        if dataset_name == "new" or dataset_name is None:
            return
        self.screen.update_Dcorr(proj_name, Dcorr)

    def get_notified(self, event_class, id, value, to_redraw = []):
        with self.lock:
            if event_class == THUMBNAIL_CLICKED:
                self.thumbnail_clicked(value, to_redraw)
            elif event_class == ABSQA_VIEW_CHANGE:
                print("CHANGE")
                self.screen.view_change(value, to_redraw)
            elif event_class == ABSQA_VERSUS_CHANGE:
                self.screen.versus_change(value, self.selected_embedding, to_redraw, self.get_dataset())
            elif event_class == ABSQA_THRESHOLD_CHANGE:
                self.screen.threshold_change(value, self.selected_embedding, to_redraw, self.get_dataset())


    def thumbnail_clicked(self, proj_name, to_redraw):
        if self.selected_proj_name != proj_name:
            self.selected_proj_name = proj_name
            embedding = self.get_embedding(proj_name)
            self.selected_embedding = embedding
            if embedding is not None:
                self.screen.projection_selected(self.selected_proj_name, embedding, to_redraw)

    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        return False
