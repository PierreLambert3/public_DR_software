from engine.gui.element import Element, Text
from utils import luminosity_change
import threading
import numpy as np
import pygame
import numba

@numba.jit(nopython=True, fastmath=True)
def shepard_dist_to_px(D_ld_raw, D_hd_raw, n_buckets, scale_with_max = False):

    sorted_LD = np.argsort(D_ld_raw)
    D_ld = D_ld_raw[sorted_LD]
    D_hd = D_hd_raw[sorted_LD]

    N = D_ld.shape[0]
    D_ld -= np.min(D_ld)
    D_hd -= np.min(D_hd)
    if scale_with_max:
        D_ld /= np.max(D_ld + 1e-8)
        D_hd /= np.max(D_hd + 1e-8)
    else:
        D_ld /= 5*np.std(D_ld + 1e-8)
        D_hd /= 5*np.std(D_hd + 1e-8)

    per_bucket = int(N/n_buckets)
    acc_hd      = np.zeros((n_buckets,))
    acc_hd_stds = np.zeros((n_buckets,))
    for i in range(n_buckets):
        L = i*per_bucket
        R = min(i*per_bucket+per_bucket, N)
        acc_hd[i]      = np.mean(D_hd[L:R])
        acc_hd_stds[i] = np.std(D_hd[L:R])

    tmp    = acc_hd.copy()
    for i in range(1, n_buckets-1):
        tmp[i] = 0.25*acc_hd[i-1] + 0.5*acc_hd[i] + 0.25*acc_hd[i+1]
    acc_hd = tmp

    within_bounds = np.ones((N,), dtype=numba.boolean)
    oob_acc = 0
    if not scale_with_max:
        for i in range(N):
            if D_ld[i] > 1 or D_hd[i] > 1:
                within_bounds[i] = 0
                oob_acc += 1

    if oob_acc/N > 0.15:
        return shepard_dist_to_px(D_ld_raw, D_hd_raw, n_buckets, scale_with_max = True)
    else:
        return D_ld, D_hd, acc_hd, acc_hd_stds*0.1, within_bounds

class Shepard_diagram(Element):
    def __init__(self, pos_pct, dim_pct, name, parent, color, uid_generator):
        super(Shepard_diagram, self).__init__(pos_pct, dim_pct, name, parent, uid = uid_generator.get(), color=color)
        self.color       = color
        self.faded_color  = luminosity_change(color, -140)
        self.very_faded_color  = luminosity_change(color, -290)
        self.vary_very_faded_color  = luminosity_change(color, -320)
        self.bright_color = luminosity_change(color, 180)
        self.smoothed   = None
        self.stds_lower = None
        self.stds_upper = None
        self.x = None
        self.y = None
        self.within_bounds = None
        self.nb_points    = 0
        self.N_buckets = 42

    def receive_distances(self, D_ld, D_hd):
        D_ld, D_hd, acc_hd, acc_hd_stds, within_bounds = shepard_dist_to_px(D_ld, D_hd, self.N_buckets)
        self.update(D_ld, D_hd, acc_hd, acc_hd_stds, within_bounds)

    def wipe(self):
        self.x = None
        self.smoothed = None

    def draw(self, screen):
        try:
            pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)
            pygame.draw.line(screen, self.color, (self.abs_pos[0], self.abs_pos[1]), (self.abs_pos[0], self.abs_pos[1]+self.dim[1]), 2)
            pygame.draw.line(screen, self.color, (self.abs_pos[0], self.abs_pos[1]+self.dim[1]), (self.abs_pos[0]+self.dim[0], self.abs_pos[1]+self.dim[1]), 2)
            if self.smoothed is not None:
                x = self.x
                y = self.y
                smoothed     = self.smoothed
                stds_lower  = self.stds_lower
                stds_upper  = self.stds_upper
                N_buckets = self.N_buckets
                n_per_bucket = int(x.shape[0]/self.N_buckets)
                half = int(n_per_bucket/2)
                for i in range(1, self.smoothed.shape[0]):
                    R = n_per_bucket*i+half
                    L = n_per_bucket*(i-1)+half
                    if stds_upper[i] > self.abs_pos[1] - self.dim[1]:
                        pygame.draw.polygon(screen, self.vary_very_faded_color, [(x[L], stds_lower[i-1]), (x[L], stds_upper[i-1]), (x[R], stds_lower[i])], 0)
                        pygame.draw.polygon(screen, self.vary_very_faded_color, [(x[L], stds_upper[i-1]), (x[R], stds_upper[i]), (x[R], stds_lower[i])], 0)

                if x is not None:
                    for i in range(1, self.nb_points):
                        if self.within_bounds[i]:
                            pygame.draw.line(screen, self.faded_color, (x[i], y[i]), (x[i]+1, y[i]+1), 2)

                for i in range(1, self.smoothed.shape[0]):
                    R = n_per_bucket*i+half
                    L = n_per_bucket*(i-1)+half
                    if smoothed[i] > self.abs_pos[1] - self.dim[1]:
                        pygame.draw.line(screen, self.bright_color, (x[L], smoothed[i-1]), (x[R], smoothed[i]), 2)
            pygame.draw.aaline(screen, (100, 0, 150), (self.abs_pos[0]+self.dim[0], self.abs_pos[1]), (self.abs_pos[0], self.abs_pos[1]+self.dim[1]), 1)
        except:
            return

    def update(self, x, y, curve, curve_stds, within_bounds):
        self.within_bounds = within_bounds
        self.smoothed    = self.abs_pos[1] + (self.dim[1] - curve * self.dim[1]).astype(int)
        self.stds_lower  = np.clip(self.abs_pos[1] + (self.dim[1] - (curve-curve_stds) * self.dim[1]).astype(int), a_min=self.abs_pos[1], a_max=self.abs_pos[1]+self.dim[1])
        self.stds_upper  = np.clip(self.abs_pos[1] + (self.dim[1] - (curve+curve_stds) * self.dim[1]).astype(int), a_min=self.abs_pos[1], a_max=self.abs_pos[1]+self.dim[1])
        self.x = (x * self.dim[0]).astype(int)
        self.x += self.abs_pos[0]
        self.y = self.abs_pos[1]
        self.y += (self.dim[1] - y * self.dim[1]).astype(int)
        self.nb_points = x.shape[0]

class Comparable_Shepard_diagram(Shepard_diagram):
    def __init__(self, pos_pct, dim_pct, name, parent, color, uid_generator, color_L, color_R):
        super(Comparable_Shepard_diagram, self).__init__(pos_pct, dim_pct, name, parent,color, uid_generator)
        self.smoothed_R   = None
        self.stds_lower_R = None
        self.stds_upper_R = None
        self.x_R = None
        self.y_R = None
        self.within_bounds_R = None
        self.color_L = luminosity_change(color_L, -200)
        self.vary_very_faded_color_L  = luminosity_change(color_L, -320)
        self.bright_color_L           = luminosity_change(color_L, 180)
        self.color_R = luminosity_change(color_R, -200)
        self.vary_very_faded_color_R  = luminosity_change(color_R, -320)
        self.bright_color_R           = luminosity_change(color_R, 180)

    def receive_distances(self, D_ld_L, D_ld_R, D_hd):
        scaled_D_ld_R, scaled_D_hd_R, acc_hd_R, acc_hd_stds_R, within_bounds_R = None, None, None, None, None
        scaled_D_ld, scaled_D_hd, acc_hd, acc_hd_stds, within_bounds = None, None, None, None, None
        if D_ld_L is not None:
            scaled_D_ld, scaled_D_hd, acc_hd, acc_hd_stds, within_bounds = shepard_dist_to_px(D_ld_L, D_hd, self.N_buckets)
        if D_ld_R is not None:
            scaled_D_ld_R, scaled_D_hd_R, acc_hd_R, acc_hd_stds_R, within_bounds_R = shepard_dist_to_px(D_ld_R, D_hd, self.N_buckets)
        if scaled_D_ld is not None or scaled_D_ld_R is not None:
            self.update(scaled_D_ld, scaled_D_ld_R, scaled_D_hd, scaled_D_hd_R, acc_hd, acc_hd_R, acc_hd_stds, acc_hd_stds_R, within_bounds, within_bounds_R)

    def wipe(self):
        self.x = None
        self.x_R = None
        self.smoothed = None

    def draw(self, screen):
        try:
            pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)
            pygame.draw.line(screen, self.color, (self.abs_pos[0], self.abs_pos[1]), (self.abs_pos[0], self.abs_pos[1]+self.dim[1]), 2)
            pygame.draw.line(screen, self.color, (self.abs_pos[0], self.abs_pos[1]+self.dim[1]), (self.abs_pos[0]+self.dim[0], self.abs_pos[1]+self.dim[1]), 2)
            # these polygons are the variance bands around the middle line
            if self.smoothed is not None or self.smoothed_R is not None:
                x = self.x
                x_R = self.x_R
                y = self.y
                y_R = self.y_R
                smoothed    = self.smoothed
                smoothed_R    = self.smoothed_R
                stds_lower  = self.stds_lower
                stds_lower_R  = self.stds_lower_R
                stds_upper  = self.stds_upper
                stds_upper_R  = self.stds_upper_R
                N_buckets = self.N_buckets
                if x is not None:
                    n_per_bucket = int(x.shape[0]/self.N_buckets)
                    N = self.smoothed.shape[0]
                elif x_R is not None:
                    n_per_bucket = int(x_R.shape[0]/self.N_buckets)
                    N = self.smoothed_R.shape[0]
                else:
                    return
                half = int(n_per_bucket/2)
                for i in range(1, N):
                    R = n_per_bucket*i+half
                    L = n_per_bucket*(i-1)+half
                    if x is not None:
                        if stds_upper[i] > self.abs_pos[1] - self.dim[1]:
                            pygame.draw.polygon(screen, self.vary_very_faded_color_L, [(x[L], stds_lower[i-1]), (x[L], stds_upper[i-1]), (x[R], stds_lower[i])], 0)
                            pygame.draw.polygon(screen, self.vary_very_faded_color_L, [(x[L], stds_upper[i-1]), (x[R], stds_upper[i]), (x[R], stds_lower[i])], 0)

                    if x_R is not None:
                        if stds_upper_R[i] > self.abs_pos[1] - self.dim[1]:
                            pygame.draw.polygon(screen, self.vary_very_faded_color_R, [(x_R[L], stds_lower_R[i-1]), (x_R[L], stds_upper_R[i-1]), (x_R[R], stds_lower_R[i])], 0)
                            pygame.draw.polygon(screen, self.vary_very_faded_color_R, [(x_R[L], stds_upper_R[i-1]), (x_R[R], stds_upper_R[i]), (x_R[R], stds_lower_R[i])], 0)

                for i in range(1, self.nb_points):
                    if x is not None and self.within_bounds[i]:
                        pygame.draw.line(screen, self.color_L, (x[i], y[i]), (x[i]+1, y[i]+1), 2)
                    if x_R is not None and self.within_bounds_R[i]:
                        pygame.draw.line(screen, self.color_R, (x_R[i], y_R[i]), (x_R[i]+1, y_R[i]+1), 2)

                for i in range(1, self.smoothed.shape[0]):
                    R = n_per_bucket*i+half
                    L = n_per_bucket*(i-1)+half
                    if x is not None:
                        if smoothed[i] > self.abs_pos[1] - self.dim[1]:
                            pygame.draw.line(screen, self.bright_color_L, (x[L], smoothed[i-1]), (x[R], smoothed[i]), 2)
                    if x_R is not None:
                        if smoothed_R[i] > self.abs_pos[1] - self.dim[1]:
                            pygame.draw.line(screen, self.bright_color_R, (x_R[L], smoothed_R[i-1]), (x_R[R], smoothed_R[i]), 2)

            pygame.draw.aaline(screen, (100, 0, 150), (self.abs_pos[0]+self.dim[0], self.abs_pos[1]), (self.abs_pos[0], self.abs_pos[1]+self.dim[1]), 1)
        except Exception as e:
            print(e)
            return

    def update(self, x, x_R, y, y_R, acc_hd, acc_hd_R, acc_hd_stds, acc_hd_stds_R, within_bounds, within_bounds_R):
        if x is not None:
            self.within_bounds = within_bounds
            self.smoothed    = self.abs_pos[1] + (self.dim[1] - acc_hd * self.dim[1]).astype(int)
            self.stds_lower  = np.clip(self.abs_pos[1] + (self.dim[1] - (acc_hd-acc_hd_stds) * self.dim[1]).astype(int), a_min=self.abs_pos[1], a_max=self.abs_pos[1]+self.dim[1])
            self.stds_upper  = np.clip(self.abs_pos[1] + (self.dim[1] - (acc_hd+acc_hd_stds) * self.dim[1]).astype(int), a_min=self.abs_pos[1], a_max=self.abs_pos[1]+self.dim[1])
            self.x = (x * self.dim[0]).astype(int)
            self.x += self.abs_pos[0]
            self.y = self.abs_pos[1]
            self.y += (self.dim[1] - y * self.dim[1]).astype(int)
            self.nb_points = x.shape[0]
        else:
            self.x = None

        if x_R is not None:
            self.stds_upper_R  = np.clip(self.abs_pos[1] + (self.dim[1] - (acc_hd_R+acc_hd_stds_R) * self.dim[1]).astype(int), a_min=self.abs_pos[1], a_max=self.abs_pos[1]+self.dim[1])
            self.within_bounds_R = within_bounds_R
            self.smoothed_R    = self.abs_pos[1] + (self.dim[1] - acc_hd_R * self.dim[1]).astype(int)
            self.stds_lower_R  = np.clip(self.abs_pos[1] + (self.dim[1] - (acc_hd_R-acc_hd_stds_R) * self.dim[1]).astype(int), a_min=self.abs_pos[1], a_max=self.abs_pos[1]+self.dim[1])
            self.x_R = (x_R * self.dim[0]).astype(int)
            self.x_R += self.abs_pos[0]
            self.y_R = self.abs_pos[1]
            self.y_R += (self.dim[1] - y_R * self.dim[1]).astype(int)
            self.nb_points = x_R.shape[0]
        else:
            self.x_R = None


class Negpos_QA_graph(Element):
    def __init__(self, pos_pct, dim_pct, name, parent, uid_generator):
        super(Negpos_QA_graph, self).__init__(pos_pct, dim_pct, name, parent, uid = uid_generator.get())
        self.listen_to(["hover"])
        self.uid_generator = uid_generator
        self.max_K     = 10
        self.px_at_K   = None #range(0, self.max_K) # the corresponding pixel associated with a given K in x axis (follows a log function)
        self.Rnx_lines = {}
        self.Rnx_raw   = {}
        self.auc_texts = {}
        self.y_px_len = self.dim[1]*0.9
        self.x_px_len = self.dim[0]*0.95
        self.anchor = (self.abs_pos[0]+3, self.abs_pos[1]+self.dim[1]-2)
        self.hovered = False
        self.mouse_y_line = None
        self.mouse_x_line = None
        self.mouse_K_text = Text((0., 0), 4, (0.3, 0.2), 16, "placeholder", self, color =  luminosity_change(self.color, -210))
        self.mouse_y_text = Text((0., 0), 4, (0.3, 0.2), 16, "placeholder", self, color =  luminosity_change(self.color, -210))
        self.min_value = -1
        self.max_value = 1

    def delete(self):
        self.px_at_K = None
        self.Rnx_lines = None
        if self.auc_texts is not None:
            for txt in self.auc_texts:
                if self.auc_texts[txt] is not None:
                    self.auc_texts[txt].delete()
            self.auc_texts = None
        self.mouse_K_text.delete()
        self.mouse_y_text.delete()
        self.mouse_K_text = None
        self.mouse_y_text = None
        super(QA_graph, self).delete()

    def remove_projection(self, proj_name):
        if proj_name in self.auc_texts:
            self.auc_texts[proj_name].delete()
            del self.auc_texts[proj_name]
            del self.Rnx_lines[proj_name]
            del self.Rnx_raw[proj_name]
            self.recompute_drawing()
            self.min_value = 1
            self.max_value = -1

    def wipe(self):
        for t in self.auc_texts:
            self.auc_texts[t].delete()
        self.Rnx_lines = {}
        self.Rnx_raw   = {}
        self.auc_texts = {}
        self.min_value = 1
        self.max_value = -1
        self.max_K     = 10

    def set_all_colors(self, color):
        for e in self.auc_texts:
            if not self.auc_texts[e].color == color:
                self.auc_texts[e].color = color
                self.auc_texts[e].txt_img = self.auc_texts[e].font.render(self.auc_texts[e].name, True, color)

    def change_selected(self, selected_name, selected_color, notselected_color):
        for e in self.auc_texts:
            if e == selected_name:
                self.auc_texts[e].color = selected_color
                self.auc_texts[e].txt_img = self.auc_texts[e].font.render(self.auc_texts[e].name, True, selected_color)
            else:
                self.auc_texts[e].color = notselected_color
                self.auc_texts[e].txt_img = self.auc_texts[e].font.render(self.auc_texts[e].name, True, notselected_color)

    def set_max_K(self, max_K):
        self.max_K   = max_K
        self.px_at_K = np.log((np.arange(max_K-1)+1))
        occupancy_ratio = self.px_at_K[-1]/self.x_px_len
        self.px_at_K = self.anchor[0]+self.px_at_K/occupancy_ratio

    def recompute_drawing(self, color_change = False):
        i = 0
        for p in self.auc_texts:
            self.auc_texts[p].abs_pos = (self.abs_pos[0] + 0.85*self.dim[0], self.abs_pos[1] + (0.06 + i*0.08)*self.dim[1])
            if color_change:
                self.auc_texts[p].recompute_img()
            i += 1

    def set_points(self, rnx, auc, proj_name, color):
        if self.max_K < 11:
            self.set_max_K(rnx.shape[0]-1)
        minn, maxx = np.min(rnx), np.max(rnx)
        update_bounds = False
        if minn < self.min_value:
            self.min_value = minn
            update_bounds = True
        if maxx > self.max_value:
            self.max_value = maxx
            update_bounds = True

        if self.deleted:
            return

        if not proj_name in self.Rnx_lines:
            self.Rnx_lines[proj_name] = np.zeros((self.max_K,))
            self.Rnx_raw[proj_name]   = rnx.copy()
        if proj_name in self.auc_texts:
            self.auc_texts[proj_name].delete()

        N_proj = len(self.Rnx_lines)
        self.auc_texts[proj_name] = Text((0.85, 0.03+0.06*N_proj), 1, (0.3, 0.2), 16, "AUC {:.3f}".format(auc), self, color = color, draw_background=True)


        bottom = -max(np.abs(self.max_value), np.abs(self.min_value))
        width  = -2*bottom
        if update_bounds:
            for key in self.Rnx_lines:
                scaled_rnx = (self.Rnx_raw[key] - bottom) / (width+1e-7)
                self.Rnx_lines[key] = self.anchor[1] - np.maximum(0, scaled_rnx*self.y_px_len)
                self.Rnx_lines[key] = self.anchor[1] - scaled_rnx*self.y_px_len
        else:
            scaled_rnx = (self.Rnx_raw[proj_name] - bottom) / (width+1e-7)
            self.Rnx_lines[proj_name] = self.anchor[1] - np.maximum(0, scaled_rnx*self.y_px_len)
            self.Rnx_lines[proj_name] = self.anchor[1] - scaled_rnx*self.y_px_len

        i = 0
        for p in self.auc_texts:
            self.auc_texts[p].abs_pos = (self.abs_pos[0] + 0.85*self.dim[0], self.abs_pos[1] + (0.06 + i*0.08)*self.dim[1])
            i += 1


    def draw(self, screen): # return True if not a hidden element
        try:
            if super(Negpos_QA_graph, self).draw(screen):
                if self.max_K < 11 or self.deleted:
                    return True
                pygame.draw.rect(screen, self.background_color, self.bounding_rect)
                pygame.draw.line(screen, self.color, (self.anchor[0], self.anchor[1]-self.y_px_len*0.5), (self.anchor[0]+self.x_px_len, self.anchor[1]-self.y_px_len*0.5), 2)
                pygame.draw.line(screen, self.color, self.anchor, (self.anchor[0], self.anchor[1]-self.y_px_len), 2)
                for proj_name in self.Rnx_lines:
                    color = self.auc_texts[proj_name].color
                    points = self.Rnx_lines[proj_name]
                    prev_point = points[0]
                    for i in range(1, max(1, points.shape[0]-10)):
                        pygame.draw.aaline(screen, color, (self.px_at_K[i-1], prev_point),  (self.px_at_K[i], points[i]), 1)
                        prev_point = points[i]
                for key in self.auc_texts:
                    self.auc_texts[key].draw(screen)
                if self.hovered:
                    # pygame.draw.line(screen, luminosity_change(self.color, -260), self.mouse_y_line[0],  self.mouse_y_line[1], 1)
                    pygame.draw.line(screen, luminosity_change(self.color, -260), self.mouse_x_line[0],  self.mouse_x_line[1], 1)
                    self.mouse_K_text.draw(screen)
            return True
        except:
            print("dataset changed while drawing graph: aborting")

    def point_is_within_graph(self, pos):
        if pos[0] > self.anchor[0] and pos[0] < self.anchor[0]+self.x_px_len:
            if pos[1] < self.anchor[1] and pos[1] > self.anchor[1]-self.y_px_len:
                return True
        return False

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.point_is_within_graph(mouse_pos) and self.max_K > 10:
            self.hovered = True
            x_pct = (mouse_pos[0]-self.anchor[0])/self.x_px_len
            log_max_K = np.log(self.max_K)
            k = int(np.exp(x_pct*log_max_K))
            rnx = (self.anchor[1] - mouse_pos[1])/self.y_px_len
            k_px = self.anchor[0]+(np.log(k)/np.log(self.max_K))*self.x_px_len
            self.schedule_awaiting(awaiting_mouse_move)
            self.mouse_y_line = ((k_px,  mouse_pos[1]), (self.anchor[0], mouse_pos[1]))
            self.mouse_x_line = ((k_px,  mouse_pos[1]), (k_px, self.anchor[1]))
            self.mouse_K_text.update(str(int(k)), (mouse_pos[0]-20, self.anchor[1]-15))
            self.mouse_y_text.update(str(round(rnx, 2)), (self.anchor[0]+4,mouse_pos[1]))
            self.schedule_draw(to_redraw)
        return True # stop parent from propagating further

    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        if not self.point_is_within_graph(mouse_positions[0]):
            self.hovered = False
            self.schedule_draw(to_redraw)
            return True
        return False

class Comparable_Negpos_QA_graph(Negpos_QA_graph):
    def __init__(self, pos_pct, dim_pct, name, parent, uid_generator):
        super(Comparable_Negpos_QA_graph, self).__init__(pos_pct, dim_pct, name, parent, uid_generator)
        self.listen_to(["hover"])

    def color_L_and_R(self, embedding_L, embedding_R, color_L, color_R, color_default):
        name_L = None
        name_R = None
        if embedding_L is not None:
            name_L = embedding_L.proj_name
        if embedding_R is not None:
            name_R = embedding_R.proj_name
        for proj_name in self.auc_texts:
            if proj_name == name_L:
                self.auc_texts[proj_name].color   = color_L
                self.auc_texts[proj_name].txt_img = self.auc_texts[proj_name].font.render(self.auc_texts[proj_name].name, True, color_L)
            elif proj_name == name_R:
                self.auc_texts[proj_name].color   = color_R
                self.auc_texts[proj_name].txt_img = self.auc_texts[proj_name].font.render(self.auc_texts[proj_name].name, True, color_R)
            else:
                self.auc_texts[proj_name].color   = color_default
                self.auc_texts[proj_name].txt_img = self.auc_texts[proj_name].font.render(self.auc_texts[proj_name].name, True, color_default)



@numba.jit(nopython=True, fastmath=True)
def relative_score_colors(scores_L, scores_R):
    N = scores_L.shape[0]
    color_left  = np.array([150, 130, 250])
    color_right = np.array([250, 150, 110])
    diff = scores_L - scores_R
    out_colors = np.zeros((N, 3))
    mu = np.mean(diff)
    for i in range(N):
        if diff[i] > 0:
            out_colors[i] = color_left*min(1., 30*(diff[i]**2))
        else:
            out_colors[i] = color_right*min(1., 30*(np.abs(diff[i])**2))
    if mu > 0:
        mu_color = color_left*min(1., 30*(mu**3))
    else:
        mu_color = color_right*min(1., 30*(np.abs(mu)**3))
    return out_colors, mu_color


@numba.jit(nopython=True, fastmath=True)
def compute_colors(scores_vrand, scores_vself, threshold):
    N = scores_vrand.shape[0]
    dark_blue     = np.array([0, 0, 240])
    bright_yellow = np.array([255, 210, 80])
    steepness     = 15 # high values = more brutal cutoff, low value = smooth

    dv_vrand = scores_vrand + 1e-8
    sigd_vrand = 1./(1+np.exp(-steepness*(dv_vrand-0.5 + (threshold-0.5))))

    dv_vself = scores_vself + 1e-8
    sigd_vself = 1./(1+np.exp(-steepness*(dv_vself-0.5 + (threshold-0.5))))

    colors_vrand = np.zeros((N, 3))
    colors_vself = np.zeros((N, 3))
    score_acc = 0.

    for i in range(N):
        score_acc += sigd_vrand[i]
        colors_vrand[i] = dark_blue*sigd_vrand[i] + bright_yellow*(1 - sigd_vrand[i])
        colors_vself[i] = dark_blue*sigd_vself[i] + bright_yellow*(1 - sigd_vself[i])
    mu_score = (score_acc/N)
    contour  = mu_score*0.5*dark_blue + (1-mu_score)*0.5*bright_yellow
    # return colors_vrand*0.5, colors_vself*0.5, contour
    return colors_vrand*0.8, colors_vself*0.8, contour



class abs_localQA_scatterplot(Element):
    def __init__(self, pos_pct, dim_pct, name, parent, X_LD, colors_vrand, colors_vself, mode, uid_generator, color, show_score = True):
        super(abs_localQA_scatterplot, self).__init__(pos_pct, dim_pct, name, parent, uid = uid_generator.get(), color=color)
        self.listen_to(["Lclick"])
        self.show_score = show_score
        self.uid_generator  = uid_generator
        self.title      = Text((0.05, 0.), 1, (0.8, 0.5), 18, name, self, draw_background=True)
        self.score_text = Text((0.85, 0.), 1, (0.5, 0.5), 18, name, self, draw_background=True)
        self.description = ""
        self.X_LD      = X_LD
        self.X_LD_px   = None
        self.colors_vrand   = colors_vrand
        self.colors_vself   = colors_vself
        self.contour        = self.color
        self.mode           = mode
        self.x_axis_length = 0.9
        self.y_axis_length = 0.9
        self.anchor = (0.01, 0.99) # bottom-left position
        self.scale_to_square()
        self.update_pos_and_dim()

    def set_title(self, new_title):
        self.title.update(new_title, None)

    def set_overall_score(self, score):
        self.score_text.update(str(score), None)

    def set_description(self, description):
        self.description = description

    def color_changes(self, scores_vrand, scores_vself, threshold):
        if scores_vrand is None or scores_vself is None:
            return
        colors_vrand, colors_vself, contour = compute_colors(scores_vrand, scores_vself, threshold)
        self.colors_vrand   = colors_vrand
        self.colors_vself   = colors_vself
        self.contour        = contour

    def set_points(self, X_LD):
        self.X_LD      = X_LD
        self.rebuild_points()

    def rebuild_points(self):
        if self.X_LD is None:
            self.X_LD_px = None
            return
        X = self.X_LD
        ax_px_len = self.x_axis_length*self.dim[0]
        ax1_min = np.min(X[:, 0])
        ax2_min = np.min(X[:, 1])
        ax1_wingspan = np.max(X[:, 0]) - ax1_min + 1e-9
        ax2_wingspan = np.max(X[:, 1]) - ax2_min + 1e-9
        x_offset = self.abs_pos[0] + self.anchor[0]*self.dim[0]
        y_offset = self.abs_pos[1] + self.anchor[1]*self.dim[1]
        X_LD_px = np.zeros(X.shape)
        idx = 0
        for obs in X:
            X_LD_px[idx][0] = ax_px_len*((obs[0]-ax1_min)/ax1_wingspan)+x_offset
            X_LD_px[idx][1] = y_offset-ax_px_len*((obs[1]-ax2_min)/ax2_wingspan)
            idx += 1
        self.X_LD_px = X_LD_px

    def scale_to_square(self):
        axis_target_length = 0.88
        yx_ratio = self.dim[1]/self.dim[0]
        if yx_ratio < 1: # we need to separate both cases because we can only scale an axis by getting it smaller (or else we would have an axis bigger that the parent container)
            self.x_axis_length = axis_target_length*yx_ratio
            self.y_axis_length = axis_target_length
        else:
            self.x_axis_length = axis_target_length
            self.y_axis_length = axis_target_length/yx_ratio
        self.rebuild_axis()
        self.rebuild_points()

    def rebuild_axis(self):
        x_axis = [self.anchor, (self.anchor[0]+self.x_axis_length, self.anchor[1]), 1, self.color]
        y_axis = [self.anchor, (self.anchor[0], self.anchor[1]-self.y_axis_length), 1, self.color]
        small_bar_x = [(self.anchor[0]+self.x_axis_length, self.anchor[1]-0.01),(self.anchor[0]+self.x_axis_length, self.anchor[1]+0.01), 1, self.color]
        small_bar_y = [(self.anchor[0]-0.01, self.anchor[1]-self.y_axis_length),(self.anchor[0]+0.01, self.anchor[1]-self.y_axis_length), 1, self.color]
        self.lines_schematics.clear()
        self.lines.clear()
        self.add_line(x_axis)
        self.add_line(y_axis)

    def update_pos_and_dim(self):
        self.dim     = (self.parent.dim[0]*self.dim_pct[0], self.parent.dim[1]*self.dim_pct[1])
        self.rel_pos = (self.parent.dim[0]*self.pos_pct[0], self.parent.dim[1]*self.pos_pct[1])
        self.abs_pos = (self.parent.abs_pos[0]+self.rel_pos[0], self.parent.abs_pos[1]+self.rel_pos[1])
        self.background_rect = pygame.Rect(self.abs_pos, self.dim)
        self.bounding_rect   = pygame.Rect((self.abs_pos[0], self.abs_pos[1]), (self.dim[0], self.dim[1]+5))
        self.scale_to_square()
        self.rebuild_texts()

    def draw(self, screen):
        if self.deleted:
            return
        try:
            if super(abs_localQA_scatterplot, self).draw(screen) and self.X_LD_px is not None:
                pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)
                pygame.draw.rect(screen, self.contour, self.bounding_rect, 3)
                N = self.X_LD.shape[0]
                coord = self.X_LD_px
                if coord is not None and self.colors_vrand is not None:
                    if self.mode == "vs. random":
                        for p_idx in range(len(coord)):
                            pygame.draw.line(screen, self.colors_vrand[p_idx], coord[p_idx], coord[p_idx], 1)
                    else:
                        for p_idx in range(len(coord)):
                            pygame.draw.line(screen, self.colors_vself[p_idx], coord[p_idx], coord[p_idx], 1)
                self.title.draw(screen)
                if self.show_score :
                    self.score_text.draw(screen)
        except:
            return True

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.point_is_inside(mouse_pos):
            print("\n")
            print(self.title.name)
            print(self.description)
            print("\n\n\n")
            return True


class rel_localQA_scatterplot(abs_localQA_scatterplot):
    def __init__(self, pos_pct, dim_pct, name, parent, X_LD, colors_vrand, colors_vself, mode, uid_generator, color):
        super(rel_localQA_scatterplot, self).__init__(pos_pct, dim_pct, name, parent,None,None,None,mode, uid_generator=uid_generator, color=color)
        self.show_score = False

    def color_changes(self, scores_L, scores_R):
        colors, contour = relative_score_colors(scores_L, scores_R)
        # colors = (np.random.uniform(size = (3 * scores_L.shape[0])) * 250).reshape((-1, 3))
        self.colors_vrand   = colors
        self.colors_vself   = colors
        self.contour        = contour

class Thumbnail(Element):
    def __init__(self, pos_pct, dim_pct, name, parent, Dcorr, RnxAUC, KNNgain, X_LD, click_listener, uid_generator, color, show_scores=True):
        super(Thumbnail, self).__init__(pos_pct, dim_pct, name, parent, uid = uid_generator.get(), color=color)
        self.listen_to(["Lclick"])
        self.uid_generator  = uid_generator
        self.click_listener = click_listener
        self.title     = Text((0.1, 0.1), 1, (0.5, 0.5), 12, name, self, draw_background=True)
        self.Dcorr     = Dcorr
        self.Dcorr_text     = None
        self.RnxAUC    = RnxAUC
        self.RnxAUC_text    = None
        self.KNNgain   = KNNgain
        self.KNNgain_text   = None
        self.X_LD      = X_LD
        self.X_LD_px   = None
        self.x_axis_length = 0.9
        self.y_axis_length = 0.9
        self.anchor = (0.01, 0.99) # bottom-left position
        self.selected = False
        self.show_scores = show_scores
        self.scale_to_square()
        self.update_pos_and_dim()

    def set_text(self, title, Dcorr, RnxAUC, KNNgain):
        if title is not None:
            if self.title is not None:
                self.title.delete()
            self.title = Text((0.05, 0.1), 1, (1, 0.5), 18, title, self, draw_background=True)
        if Dcorr is not None:
            self.Dcorr = Dcorr
            if self.Dcorr_text is not None:
                self.Dcorr_text.delete()
            self.Dcorr_text = Text((0.6, 0.3), 1, (0.5, 0.5), 16, "Dcorr "+str(Dcorr), self, draw_background=True)
        if RnxAUC is not None:
            self.RnxAUC = RnxAUC
            if self.RnxAUC_text is not None:
                self.RnxAUC_text.delete()
            self.RnxAUC_text = Text((0.6, 0.5), 1, (0.5, 0.5), 16, "Rnx   "+str(RnxAUC), self, draw_background=True)
        if KNNgain is not None:
            self.KNNgain = KNNgain
            if self.KNNgain_text is not None:
                self.KNNgain_text.delete()
            self.KNNgain_text = Text((0.6, 0.7), 1, (0.5, 0.5), 16, "KNN  "+str(KNNgain), self, draw_background=True)


    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.click_listener is not None:
            self.click_listener.notify(self.name, to_redraw)
        return True

    def set_points(self, X_LD):
        self.X_LD      = X_LD
        self.rebuild_points()

    def rebuild_points(self):
        if self.X_LD is None:
            self.X_LD_px = None
            return
        X = self.X_LD
        ax_px_len = self.x_axis_length*self.dim[0]
        ax1_min = np.min(X[:, 0])
        ax2_min = np.min(X[:, 1])
        ax1_wingspan = np.max(X[:, 0]) - ax1_min + 1e-9
        ax2_wingspan = np.max(X[:, 1]) - ax2_min + 1e-9
        x_offset = self.abs_pos[0] + self.anchor[0]*self.dim[0]
        y_offset = self.abs_pos[1] + self.anchor[1]*self.dim[1]
        X_LD_px = np.zeros(X.shape)
        idx = 0
        for obs in X:
            X_LD_px[idx][0] = ax_px_len*((obs[0]-ax1_min)/ax1_wingspan)+x_offset
            X_LD_px[idx][1] = y_offset-ax_px_len*((obs[1]-ax2_min)/ax2_wingspan)
            idx += 1
        self.X_LD_px = X_LD_px

    def scale_to_square(self):
        axis_target_length = 0.88
        yx_ratio = self.dim[1]/self.dim[0]
        if yx_ratio < 1: # we need to separate both cases because we can only scale an axis by getting it smaller (or else we would have an axis bigger that the parent container)
            self.x_axis_length = axis_target_length*yx_ratio
            self.y_axis_length = axis_target_length
        else:
            self.x_axis_length = axis_target_length
            self.y_axis_length = axis_target_length/yx_ratio
        self.rebuild_axis()
        self.rebuild_points()

    def rebuild_axis(self):
        x_axis = [self.anchor, (self.anchor[0]+self.x_axis_length, self.anchor[1]), 1, self.color]
        y_axis = [self.anchor, (self.anchor[0], self.anchor[1]-self.y_axis_length), 1, self.color]
        small_bar_x = [(self.anchor[0]+self.x_axis_length, self.anchor[1]-0.01),(self.anchor[0]+self.x_axis_length, self.anchor[1]+0.01), 1, self.color]
        small_bar_y = [(self.anchor[0]-0.01, self.anchor[1]-self.y_axis_length),(self.anchor[0]+0.01, self.anchor[1]-self.y_axis_length), 1, self.color]
        self.lines_schematics.clear()
        self.lines.clear()
        self.add_line(x_axis)
        self.add_line(y_axis)

    def update_pos_and_dim(self):
        self.dim     = (self.parent.dim[0]*self.dim_pct[0], self.parent.dim[1]*self.dim_pct[1])
        self.rel_pos = (self.parent.dim[0]*self.pos_pct[0], self.parent.dim[1]*self.pos_pct[1])
        self.abs_pos = (self.parent.abs_pos[0]+self.rel_pos[0], self.parent.abs_pos[1]+self.rel_pos[1])
        self.background_rect = pygame.Rect(self.abs_pos, self.dim)
        self.bounding_rect   = pygame.Rect((self.abs_pos[0], self.abs_pos[1]), (self.dim[0], self.dim[1]+5))
        self.scale_to_square()
        self.rebuild_texts()

    def delete(self):
        if self.click_listener is not None:
            self.click_listener.delete()
            self.click_listener = None
        super(Thumbnail, self).delete()

    def draw(self, screen):
        if self.deleted:
            return
        try:
            pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)
            if self.selected:
                pygame.draw.rect(screen, self.color, pygame.Rect(self.lines[1][1], (self.x_axis_length*self.dim[0], self.y_axis_length*self.dim[1])), 3)
            if super(Thumbnail, self).draw(screen) and self.X_LD_px is not None:
                N = self.X_LD.shape[0]
                coord = self.X_LD_px
                if coord is not None:
                    for p in coord:
                        pygame.draw.line(screen, self.color, p, p, 1)
                if self.title is not None:
                    self.title.draw(screen)
                if self.show_scores:
                    if self.Dcorr_text is not None:
                        self.Dcorr_text.draw(screen)
                    if self.RnxAUC_text is not None:
                        self.RnxAUC_text.draw(screen)
                    if self.KNNgain_text is not None:
                        self.KNNgain_text.draw(screen)
        except:
            return True

class QA_graph(Element):
    def __init__(self, pos_pct, dim_pct, name, parent, uid_generator,  auto_set_max_k=False):
        super(QA_graph, self).__init__(pos_pct, dim_pct, name, parent, uid = uid_generator.get())
        self.listen_to(["hover"])
        self.uid_generator = uid_generator
        self.auto_set_max_k = auto_set_max_k
        self.max_K     = 10
        self.px_at_K   = None #range(0, self.max_K) # the corresponding pixel associated with a given K in x axis (follows a log function)
        self.Rnx_lines = {}
        self.auc_texts = {}
        self.y_px_len = self.dim[1]*0.9
        self.x_px_len = self.dim[0]*0.95
        self.anchor = (self.abs_pos[0]+3, self.abs_pos[1]+self.dim[1]-2)
        self.hovered = False
        self.mouse_y_line = None
        self.mouse_x_line = None
        self.mouse_K_text = Text((0., 0), 4, (0.3, 0.2), 16, "placeholder", self, color =  luminosity_change(self.color, -210))
        self.mouse_y_text = Text((0., 0), 4, (0.3, 0.2), 16, "placeholder", self, color =  luminosity_change(self.color, -210))

    def delete(self):
        self.px_at_K = None
        self.Rnx_lines = None
        if self.auc_texts is not None:
            for txt in self.auc_texts:
                if self.auc_texts[txt] is not None:
                    self.auc_texts[txt].delete()
            self.auc_texts = None
        self.mouse_K_text.delete()
        self.mouse_y_text.delete()
        self.mouse_K_text = None
        self.mouse_y_text = None
        super(QA_graph, self).delete()

    def change_selected(self, selected_name, selected_color, notselected_color):
        for e in self.auc_texts:
            if e == selected_name:
                self.auc_texts[e].color = selected_color
                try:
                    self.auc_texts[e].txt_img = self.auc_texts[e].font.render(self.auc_texts[e].name, True, selected_color)
                except:
                    return
            else:
                self.auc_texts[e].color = notselected_color
                try:
                    self.auc_texts[e].txt_img = self.auc_texts[e].font.render(self.auc_texts[e].name, True, notselected_color)
                except:
                    return

    def remove_projection(self, proj_name):
        if proj_name in self.auc_texts:
            self.auc_texts[proj_name].delete()
            del self.auc_texts[proj_name]
            del self.Rnx_lines[proj_name]
            self.recompute_drawing()

    def wipe(self):
        for t in self.auc_texts:
            self.auc_texts[t].delete()
        self.Rnx_lines = {}
        self.auc_texts = {}
        self.max_K = 10

    def set_all_colors(self, color):
        for e in self.auc_texts:
            if not self.auc_texts[e].color == color:
                self.auc_texts[e].color = color
                self.auc_texts[e].txt_img = self.auc_texts[e].font.render(self.auc_texts[e].name, True, color)

    def set_max_K(self, max_K):
        self.max_K   = max_K
        self.px_at_K = np.log((np.arange(max_K-1)+1))
        occupancy_ratio = self.px_at_K[-1]/self.x_px_len
        self.px_at_K = self.anchor[0]+self.px_at_K/occupancy_ratio

    def recompute_drawing(self, color_change = False):
        i = 0
        for p in self.auc_texts:
            self.auc_texts[p].abs_pos = (self.abs_pos[0] + 0.85*self.dim[0], self.abs_pos[1] + (0.06 + i*0.08)*self.dim[1])
            if color_change:
                self.auc_texts[p].recompute_img()
            i += 1

    def set_points(self, rnx, auc, proj_name, color):
        if self.deleted:
            return
        if self.auto_set_max_k:
            self.set_max_K(rnx.shape[0]-1)
        if not proj_name in self.Rnx_lines:
            self.Rnx_lines[proj_name] = np.zeros((self.max_K,))
        if proj_name in self.auc_texts:
            self.auc_texts[proj_name].delete()
        N_proj = len(self.Rnx_lines)
        self.auc_texts[proj_name] = Text((0.85, 0.03+0.06*N_proj), 1, (0.3, 0.2), 16, "AUC {:.3f}".format(auc), self, color = color, draw_background=True)
        self.Rnx_lines[proj_name] = self.anchor[1] - np.maximum(0, rnx*self.y_px_len) # i had an rnx around -2e-6 once:a numerical error introduced by floats?
        self.Rnx_lines[proj_name] = self.anchor[1] - rnx*self.y_px_len # i had an rnx around -2e-6 once:a numerical error introduced by floats?

        i = 0
        for p in self.auc_texts:
            self.auc_texts[p].abs_pos = (self.abs_pos[0] + 0.85*self.dim[0], self.abs_pos[1] + (0.06 + i*0.08)*self.dim[1])
            i += 1


    def draw(self, screen): # return True if not a hidden element
        try:
            if super(QA_graph, self).draw(screen):
                if self.max_K < 11 or self.deleted:
                    return True
                pygame.draw.rect(screen, self.background_color, self.bounding_rect)
                pygame.draw.line(screen, self.color, self.anchor, (self.anchor[0]+self.x_px_len, self.anchor[1]), 2)
                pygame.draw.line(screen, self.color, self.anchor, (self.anchor[0], self.anchor[1]-self.y_px_len), 2)
                for proj_name in self.Rnx_lines:
                    color = self.auc_texts[proj_name].color
                    points = self.Rnx_lines[proj_name]
                    prev_point = points[0]
                    for i in range(1, max(1, points.shape[0]-10)):
                        pygame.draw.aaline(screen, color, (self.px_at_K[i-1], prev_point),  (self.px_at_K[i], points[i]), 1)
                        prev_point = points[i]
                for key in self.auc_texts:
                    self.auc_texts[key].draw(screen)
                if self.hovered:
                    pygame.draw.line(screen, luminosity_change(self.color, -260), self.mouse_y_line[0],  self.mouse_y_line[1], 1)
                    pygame.draw.line(screen, luminosity_change(self.color, -260), self.mouse_x_line[0],  self.mouse_x_line[1], 1)
                    self.mouse_K_text.draw(screen)
                    self.mouse_y_text.draw(screen)
            return True
        except:
            print("dataset changed while drawing graph: aborting")

    def point_is_within_graph(self, pos):
        if pos[0] > self.anchor[0] and pos[0] < self.anchor[0]+self.x_px_len:
            if pos[1] < self.anchor[1] and pos[1] > self.anchor[1]-self.y_px_len:
                return True
        return False

    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.point_is_within_graph(mouse_pos) and self.max_K > 10:
            self.hovered = True
            x_pct = (mouse_pos[0]-self.anchor[0])/self.x_px_len
            log_max_K = np.log(self.max_K)
            k = int(np.exp(x_pct*log_max_K))
            rnx = (self.anchor[1] - mouse_pos[1])/self.y_px_len
            k_px = self.anchor[0]+(np.log(k)/np.log(self.max_K))*self.x_px_len
            self.schedule_awaiting(awaiting_mouse_move)
            self.mouse_y_line = ((k_px,  mouse_pos[1]), (self.anchor[0], mouse_pos[1]))
            self.mouse_x_line = ((k_px,  mouse_pos[1]), (k_px, self.anchor[1]))
            self.mouse_K_text.update(str(int(k)), (mouse_pos[0]-20, self.anchor[1]-15))
            self.mouse_y_text.update(str(round(rnx, 2)), (self.anchor[0]+4,mouse_pos[1]))
            self.schedule_draw(to_redraw)
        return True # stop parent from propagating further

    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        if not self.point_is_within_graph(mouse_positions[0]):
            self.hovered = False
            self.schedule_draw(to_redraw)
            return True
        return False



class Comparable_QA_graph(QA_graph):
    def __init__(self, pos_pct, dim_pct, name, parent, uid_generator):
        super(Comparable_QA_graph, self).__init__(pos_pct, dim_pct, name, parent, uid_generator, True)
        self.listen_to(["hover"])

    def color_L_and_R(self, embedding_L, embedding_R, color_L, color_R, color_default):
        name_L = None
        name_R = None
        if embedding_L is not None:
            name_L = embedding_L.proj_name
        if embedding_R is not None:
            name_R = embedding_R.proj_name
        for proj_name in self.auc_texts:
            if proj_name == name_L:
                self.auc_texts[proj_name].color   = color_L
                self.auc_texts[proj_name].txt_img = self.auc_texts[proj_name].font.render(self.auc_texts[proj_name].name, True, color_L)
            elif proj_name == name_R:
                self.auc_texts[proj_name].color   = color_R
                self.auc_texts[proj_name].txt_img = self.auc_texts[proj_name].font.render(self.auc_texts[proj_name].name, True, color_R)
            else:
                self.auc_texts[proj_name].color   = color_default
                self.auc_texts[proj_name].txt_img = self.auc_texts[proj_name].font.render(self.auc_texts[proj_name].name, True, color_default)
