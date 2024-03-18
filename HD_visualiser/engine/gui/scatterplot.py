from engine.gui.element import Element, Text
from engine.gui.selector import Button
from engine.gui.listener import Listener
from engine.gui.event_ids import *
import pygame
import numpy as np
import threading
from matplotlib.path import Path

class Scatterplot(Element):
    def __init__(self, pos_pct, dim_pct, dataset_name, proj_name, parent, uid_generator, is_labeled, is_classification, manager, color):
        self.ctrl_pressed = False
        self.dataset_name = dataset_name
        self.proj_name    = proj_name
        super(Scatterplot, self).__init__(pos_pct, dim_pct,  dataset_name+"___"+proj_name, parent, uid=uid_generator.get(), color=color)
        self.listen_to(["Lclick","Rclick","hover"])
        self.uid_generator = uid_generator
        self.anchor = (0.01, 0.99) # bottom-left position
        self.x_axis_length = 0.88
        self.y_axis_length = 0.88
        close_listener = Listener(SCATTERPLOT_CLOSED, [manager])
        close_listener.misc_info = self.name
        save_listener  = Listener(SCATTERPLOT_SAVE, [manager])
        save_listener.misc_info = self.name
        self.close_button = Button((0., 0), (0.15, 0.15), "X", self, uid_generator, on_value_change_listener=close_listener, color=np.array([240, 30, 10]), background_color=np.array([40, 0, 0]))
        self.save_button  = Button((0.15, 0), (0.15, 0.15), "save", self, uid_generator, on_value_change_listener=save_listener, color=np.array([10, 240, 10]), background_color=np.array([0, 40, 0]))
        self.close_button.disable()
        self.save_button.enable()

        self.KNN_selection_listener     = Listener(SCATTERPLOT_KNN_SELECTION, [manager])
        self.drawing_selection_listener = Listener(SCATTERPLOT_DRAW_SELECTION, [manager])
        self.right_click_listener       = Listener(SCATTERPLOT_RIGHT_CLICK, [manager])
        self.on_selected_listener       = Listener(SCATTERPLOT_SELECTED, [manager])
        self.update_heatmap_listener    = Listener(UPDATE_HEATMAP, [manager])

        self.add_text([(0.3, 0.07), 2, (0.8,0.3), 18, self.proj_name, self.color])
        self.is_labeled = is_labeled
        self.is_classification = is_classification
        self.selected  = False
        self.heatmap   = None
        self.converged = False
        self.X_LD      = None
        self.Y         = None
        self.X_LD_px   = None
        self.px_to_LD_offsets = None
        self.px_to_LD_coefs   = None
        self.selected_points = None
        self.scale_to_square()
        self.update_pos_and_dim()
        self.drawing = False
        self.draw_heatmap = False
        self.draw_grid = np.zeros((150, 150), dtype = int)
        self.draw_trajectory = []
        self.drawing_start = (None, None)
        self.last_draw     = (None, None)
        self.selected_points_from_drawing = np.array([], dtype=int)
        self.lock = threading.Lock()

    def delete(self):
        with self.lock:
            if self.heatmap is not None:
                self.heatmap.delete()
                self.heatmap = None
            if self.KNN_selection_listener is not None:
                self.KNN_selection_listener.delete()
                self.KNN_selection_listener = None
            if self.drawing_selection_listener is not None:
                self.drawing_selection_listener.delete()
                self.drawing_selection_listener = None
            if self.right_click_listener is not None:
                self.right_click_listener.delete()
                self.right_click_listener = None
            if self.close_button is not None:
                self.close_button.delete()
                self.close_button = None
            if self.save_button is not None:
                self.save_button.delete()
                self.save_button = None
            super(Scatterplot, self).delete()

    def update_color(self, color):
        self.color = color
        for t in self.texts:
            t.delete()
        self.texts = []
        self.texts_schematics = []
        self.add_text([(0.3, 0.07), 2, (0.8,0.3), 18, self.proj_name, self.color])

    def set_points(self, X_LD, Y, Y_colors):
        self.px_to_LD_coefs   = None
        self.px_to_LD_offsets = None
        self.X_LD      = X_LD
        self.Y         = Y
        self.Y_colors  = Y_colors
        self.rebuild_points()

    def px_pos_to_LD(self, mouse_pos):
        if self.X_LD is None or self.px_to_LD_coefs is None:
            return
        return np.array([mouse_pos[0]*self.px_to_LD_coefs[0] + self.px_to_LD_offsets[0], mouse_pos[1]*self.px_to_LD_coefs[1] + self.px_to_LD_offsets[1]]).reshape((1, 2))

    def rebuild_points(self):
        if self.X_LD is None:
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

        self.px_to_LD_offsets = (-x_offset*ax1_wingspan/ax_px_len +ax1_min, y_offset*ax2_wingspan/ax_px_len +ax2_min)
        self.px_to_LD_coefs   = (ax1_wingspan/ax_px_len, -ax2_wingspan/ax_px_len)


    def scale_to_square(self):
        self.px_to_LD_coefs   = None
        self.px_to_LD_offsets = None
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
        self.add_line(small_bar_y)
        self.add_line(small_bar_x)

    def update_pos_and_dim(self):
        self.dim     = (self.parent.dim[0]*self.dim_pct[0], self.parent.dim[1]*self.dim_pct[1])
        self.rel_pos = (self.parent.dim[0]*self.pos_pct[0], self.parent.dim[1]*self.pos_pct[1])
        self.abs_pos = (self.parent.abs_pos[0]+self.rel_pos[0], self.parent.abs_pos[1]+self.rel_pos[1])
        self.background_rect = pygame.Rect(self.abs_pos, self.dim)
        self.bounding_rect   = pygame.Rect((self.abs_pos[0], self.abs_pos[1]), (self.dim[0], self.dim[1]+5))
        self.scale_to_square()
        self.rebuild_texts()
        self.close_button.update_pos_and_dim()
        self.save_button.update_pos_and_dim()

    def draw(self, screen):
        if self.deleted:
            return
        try:
            pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)
            if self.selected:
                pygame.draw.rect(screen, self.color, pygame.Rect(self.lines[1][1], (self.x_axis_length*self.dim[0], self.y_axis_length*self.dim[1])), 3)
            if super(Scatterplot, self).draw(screen) and self.X_LD_px is not None:
                N = self.X_LD.shape[0]
                coord = self.X_LD_px
                if coord is not None:
                    if coord.shape[0] > 1400:
                        thickness = 2
                    elif coord.shape[0] > 600:
                        thickness = 3
                    else:
                        thickness = 4
                    if self.is_labeled:
                        if thickness == 2:
                            for i in range(0, N):
                                pygame.draw.line(screen, self.Y_colors[i], coord[i], coord[i], thickness)
                        else:
                            for i in range(0, N):
                                pygame.draw.line(screen, self.Y_colors[i], coord[i], (coord[i,0], coord[i,1]+1), thickness)
                                pygame.draw.line(screen, self.Y_colors[i], (coord[i,0]-1, coord[i,1]), coord[i], thickness)
                    else:
                        for p in coord:
                            pygame.draw.line(screen, self.color, p, p, thickness)
                    if self.selected_points is not None:
                        for p in self.selected_points:
                            pygame.draw.line(screen, (240, 230, 255), (coord[p][0],coord[p][1]-1), (coord[p][0],coord[p][1]+1), 2 )
                    if len(self.draw_trajectory) > 2:
                        grid_res, _ = self.draw_grid.shape
                        prev_x = self.abs_pos[0]+self.dim[0]*self.anchor[0]+  self.dim[0]*self.x_axis_length*(self.draw_trajectory[0][0]/grid_res)
                        prev_y = self.abs_pos[1]+self.dim[1]*(self.anchor[1]-self.y_axis_length) + self.y_axis_length*self.dim[1]*(self.draw_trajectory[0][1]/grid_res)
                        for i in range(1, len(self.draw_trajectory)):
                            now_x = self.abs_pos[0]+self.dim[0]*self.anchor[0]+  self.dim[0]*self.x_axis_length*(self.draw_trajectory[i][0]/grid_res)
                            now_y = self.abs_pos[1]+self.dim[1]*(self.anchor[1]-self.y_axis_length) + self.y_axis_length*self.dim[1]*(self.draw_trajectory[i][1]/grid_res)
                            pygame.draw.line(screen, (255, 102, 20), (prev_x,prev_y), (now_x,now_y), 2)
                            prev_x, prev_y = now_x, now_y
            if not self.close_button.hidden:
                self.close_button.draw(screen)
                self.save_button.draw(screen)
            if (not self.drawing) and self.draw_heatmap:
                self.heatmap.draw(screen)
        except Exception as e: # if the scatterplot gets deleted and a model tries to draw in it
            if self.close_button is not None:
                self.close_button.on_value_change.notify(True, [])

    def point_is_within_plot(self, pos):
        x_relative = pos[0]-self.abs_pos[0]-self.anchor[0]*self.dim[0]
        y_relative = -(pos[1]-self.abs_pos[1]-self.anchor[1]*self.dim[1])
        if x_relative > 0 and x_relative < self.x_axis_length*self.dim[0]:
            if y_relative > 0 and y_relative < self.y_axis_length*self.dim[1]:
                return True
        return False

    def compute_which_points_are_inside(self):
        grid_res, _ = self.draw_grid.shape
        grid = np.zeros((grid_res*grid_res, 2), dtype=int)
        cnt = 0
        for r in range(grid_res): # TODO: should probably make this cleaner with meshgrid
            for c in range(grid_res):
                grid[cnt, 0] = r
                grid[cnt, 1] = c
                cnt += 1
        inside_or_outside = Path(self.draw_trajectory).contains_points(grid).reshape((grid_res, grid_res))

        valid = []
        for obs in range(self.X_LD_px.shape[0]):
            x_pct = min(0.999, (self.X_LD_px[obs, 0]-self.abs_pos[0]-self.dim[0]*self.anchor[0]) / (self.x_axis_length*self.dim[0]))
            y_pct = min(0.999, (self.X_LD_px[obs, 1]-self.abs_pos[1]-self.dim[1]*(self.anchor[1]-self.y_axis_length)) / (self.y_axis_length*self.dim[1]))
            coord = int(x_pct*grid_res), int(y_pct*grid_res)
            if inside_or_outside[coord]:
                valid.append(obs)
        self.selected_points_from_drawing = np.array(valid)

    def close_drawing(self, release_pos):
        if len(self.draw_trajectory) < 3:
            self.wipe_drawing()
            return
        first_x, first_y = self.draw_trajectory[0]
        last_x, last_y   = self.draw_trajectory[-1]
        dx, dy = first_x - last_x, first_y - last_y
        steps = abs(dx) + abs(dy)
        if steps > 0:
            dx /= steps
            dy /= steps
            for i in range(steps):
                self.draw_grid[int(last_x + i*dx), int(last_y + i*dy)] = 1
        self.draw_trajectory.append((first_x, first_y))
        self.compute_which_points_are_inside()

    def add_drawing_point(self, pos, to_redraw):
        grid_res, _ = self.draw_grid.shape
        x_pct = min(0.998, max(0., (pos[0]-self.abs_pos[0]-self.dim[0]*self.anchor[0]) / (self.x_axis_length*self.dim[0]) ))
        y_pct = min(0.998, max(0., (pos[1]-self.abs_pos[1]-self.dim[1]*(self.anchor[1]-self.y_axis_length)) / (self.y_axis_length*self.dim[1]) ))
        coordinates = int(x_pct*grid_res), int(y_pct*grid_res)
        if len(self.draw_trajectory) > 1:
            prev_x, prev_y = self.draw_trajectory[-1]
            dx, dy = prev_x - coordinates[0], prev_y -  coordinates[1]
            steps = abs(dx) + abs(dy)
            if steps > 0:
                dx /= steps
                dy /= steps
                for i in range(steps):
                    self.draw_grid[int(coordinates[0] + i*dx), int(coordinates[1] + i*dy)] = 1
        self.draw_trajectory.append(coordinates)
        self.draw_grid[coordinates] = 1
        self.last_draw = coordinates
        if self.drawing_start[0] is None:
            self.drawing_start = self.last_draw
        self.schedule_draw(to_redraw)

    def wipe_drawing(self):
        self.draw_grid.fill(0)
        self.draw_trajectory = []
        self.drawing_start = (None, None)

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.ctrl_pressed:
            if self.close_button.point_is_inside(mouse_pos):
                return self.close_button.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release)
            elif self.save_button.point_is_inside(mouse_pos) and self.converged:
                return self.save_button.propagate_Lmouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release)
            elif self.point_is_within_plot(mouse_pos):
                self.drawing = True
                self.schedule_awaiting(awaiting_mouse_release)
                self.schedule_awaiting(awaiting_mouse_move)
        else:
            if not self.selected:
                self.on_selected_listener.notify((self.dataset_name, self.proj_name), to_redraw)
                return True
            else: # on the selected scatterplot
                if self.point_is_within_plot(mouse_pos):
                    self.beeing_dragged = True
                    self.drawing = False
                    self.wipe_drawing()
                    self.schedule_awaiting(awaiting_mouse_release)
                    self.schedule_awaiting(awaiting_mouse_move)
                    self.KNN_selection_listener.notify((self.px_pos_to_LD(mouse_pos), self.dataset_name, self.proj_name), to_redraw)
        return True

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.point_is_within_plot(mouse_pos):
            self.right_click_listener.notify((mouse_pos, windows), to_redraw)
            return True
        return False


    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.ctrl_pressed:
            if self.close_button.point_is_inside(mouse_pos):
                return self.close_button.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move)
            elif self.save_button.point_is_inside(mouse_pos):
                return self.save_button.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move)
            elif self.point_is_within_plot(mouse_pos):
                self.update_heatmap_listener.notify((mouse_pos, self.px_pos_to_LD(mouse_pos), self.dataset_name, self.proj_name), to_redraw)
                self.draw_heatmap = True
        return True

    def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
        if self.drawing:
            self.close_drawing(release_pos)
            self.schedule_draw(to_redraw)
            self.drawing_selection_listener.notify((self.selected_points_from_drawing, self.dataset_name, self.proj_name), to_redraw)
        self.beeing_dragged = False
        self.drawing = False
        return True

    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        if self.stop_awaiting_hover:
            self.stop_awaiting_hover = False
            return True
        # check if still beeing clicked: else remove self from awaiting_mouse_move
        if self.drawing:
            self.add_drawing_point(mouse_positions[0], to_redraw)
            return False
        elif self.beeing_dragged:
            self.KNN_selection_listener.notify((self.px_pos_to_LD(mouse_positions[0]), self.dataset_name, self.proj_name), to_redraw)
            return False
        return True

    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        print("awaited key press: ", pressed_keys)
        return True










# class Scatterplot(Element):
#     def __init__(self, pos_pct, dim_pct, name, parent, uid_generator, on_selected_listener, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None, on_click_start_listener=None, on_drag_listener=None, close_btn_listener=None, heatmap = None):
#         self.on_selected_listener = on_selected_listener
#         self.notify_on_value_change = False # value change here is when the selected points are changed
#         self.on_value_change = on_value_change_listener
#         self.notify_on_click_start = False # used for dragging
#         self.on_click_start = on_click_start_listener
#         self.on_drag_listener = on_drag_listener
#         self.converged = False
#         self.can_be_selected = False
#         super(Scatterplot, self).__init__(pos_pct, dim_pct, name, parent,uid_generator=uid_generator, on_Lclick_listener=on_Lclick_listener, on_Rclick_listener=on_Rclick_listener, on_hover_listener=on_hover_listener, on_scroll_listener=on_scroll_listener)
#         self.notify_on_hover = False
#         self.close_button = vf.Button((0., 0), (0.05, 0.05), "x", self, on_value_change_listener=close_btn_listener)
#         self.close_button.disable()
#         self.background_rect = pygame.Rect(self.abs_pos, self.dim)
#         self.bounding_rect = pygame.Rect((self.abs_pos[0], self.abs_pos[1]), (self.dim[0], self.dim[1]+5))
#         self.anchor = (0.01, 0.99) # bottom-left position
#         self.x_axis_length = 0.88
#         self.y_axis_length = 0.88
#         self.coordinates  = None # the pixel coordinates of the points of projection
#         self.converging_view = None
#         self.scale_to_square()
#         # Y things, ignored if not supervised dataset
#         self.is_labeled = False
#         self.is_classification = False
#         self.Y_colors = []
#         self.min_max_Y = (1, 1)
#         # interactivity things:
#         self.selected = False
#         self.selected_points    = []   # list of selected point (points become selected after user clicked on some points. selection strategy can vary)
#         self.KNN_selection_params   = {"K": 10} # click one point, select the KNN points
#         self.beeing_dragged = False
#         self.drawing        = False
#
#         self.heatmap = heatmap
#         self.draw_heatmap = False
#
#         # drawing things
#         self.draw_grid = np.zeros((150, 150), dtype = int)
#         self.draw_trajectory = []
#         self.drawing_start = (None, None)
#         self.last_draw     = (None, None)
#         self.selected_from_draw = False
#
#     def delete(self):
#         if self.on_value_change is not None:
#             self.on_value_change.delete()
#             self.on_value_change  = None
#         if self.on_click_start is not None:
#             self.on_click_start.delete()
#             self.on_click_start   = None
#         if self.on_drag_listener is not None:
#             self.on_drag_listener.delete()
#             self.on_drag_listener = None
#         if self.right_click_window is not None:
#             self.right_click_window.delete()
#             self.right_click_window = None
#         if self.close_button is not None:
#             self.close_button.delete()
#             self.close_button = None
#         self.background_rect = None
#         self.bounding_rect   = None
#         if self.converging_view is not None:
#             self.converging_view.delete()
#             self.converging_view = None
#         self.Y_colors = None
#         self.selected_points = None
#         if self.heatmap is not None:
#             self.heatmap.delete()
#             self.heatmap = None
#         super(Scatterplot, self).delete()
#
#     def compute_which_points_are_inside(self):
#         grid_res, _ = self.draw_grid.shape
#         grid = np.zeros((grid_res*grid_res, 2), dtype=int)
#         cnt = 0
#         for r in range(grid_res): # TODO: should probably make this cleaner with meshgrid
#             for c in range(grid_res):
#                 grid[cnt, 0] = r
#                 grid[cnt, 1] = c
#                 cnt += 1
#         inside_or_outside = matplotlib.path.Path(self.draw_trajectory).contains_points(grid).reshape((grid_res, grid_res))
#
#         valid = []
#         for obs in range(self.coordinates.shape[0]):
#             x_pct = (self.coordinates[obs, 0]-self.abs_pos[0]-self.dim[0]*self.anchor[0]) / (self.x_axis_length*self.dim[0])
#             y_pct = (self.coordinates[obs, 1]-self.abs_pos[1]-self.dim[1]*(self.anchor[1]-self.y_axis_length)) / (self.y_axis_length*self.dim[1])
#             coord = int(x_pct*grid_res), int(y_pct*grid_res)
#             if inside_or_outside[coord]:
#                 valid.append(obs)
#         self.selected_points    = valid
#         self.selected_from_draw = True
#
#     def get_subset(self): #gets the points within the drawn polygon
#         return self.selected_points
#
#     def wipe_drawing(self):
#         self.draw_grid.fill(0)
#         self.draw_trajectory = []
#         self.drawing_start = (None, None)
#
#     def close_drawing(self, release_pos):
#         if len(self.draw_trajectory) < 3:
#             self.wipe_drawing()
#             return
#         first_x, first_y = self.draw_trajectory[0]
#         last_x, last_y   = self.draw_trajectory[-1]
#         dx, dy = first_x - last_x, first_y - last_y
#         steps = abs(dx) + abs(dy)
#         if steps > 0:
#             dx /= steps
#             dy /= steps
#             for i in range(steps):
#                 self.draw_grid[int(last_x + i*dx), int(last_y + i*dy)] = 1
#         self.draw_trajectory.append((first_x, first_y))
#         self.compute_which_points_are_inside()
#
#     def add_drawing_point(self, pos, to_redraw):
#         grid_res, _ = self.draw_grid.shape
#         x_pct = min(0.998, max(0., (pos[0]-self.abs_pos[0]-self.dim[0]*self.anchor[0]) / (self.x_axis_length*self.dim[0]) ))
#         y_pct = min(0.998, max(0., (pos[1]-self.abs_pos[1]-self.dim[1]*(self.anchor[1]-self.y_axis_length)) / (self.y_axis_length*self.dim[1]) ))
#         coordinates = int(x_pct*grid_res), int(y_pct*grid_res)
#         if len(self.draw_trajectory) > 1:
#             prev_x, prev_y = self.draw_trajectory[-1]
#             dx, dy = prev_x - coordinates[0], prev_y -  coordinates[1]
#             steps = abs(dx) + abs(dy)
#             if steps > 0:
#                 dx /= steps
#                 dy /= steps
#                 for i in range(steps):
#                     self.draw_grid[int(coordinates[0] + i*dx), int(coordinates[1] + i*dy)] = 1
#
#         self.draw_trajectory.append(coordinates)
#         self.draw_grid[coordinates] = 1
#         self.last_draw = coordinates
#         if self.drawing_start[0] is None:
#             self.drawing_start = self.last_draw
#         self.schedule_draw(to_redraw)
#
#     def point_is_within_plot(self, pos):
#         x_relative = pos[0]-self.abs_pos[0]-self.anchor[0]*self.dim[0]
#         y_relative = -(pos[1]-self.abs_pos[1]-self.anchor[1]*self.dim[1])
#         if x_relative > 0 and x_relative < self.x_axis_length*self.dim[0]:
#             if y_relative > 0 and y_relative < self.y_axis_length*self.dim[1]:
#                 return True
#         return False
#
#     def px_to_Z_space(self, pos):
#         x_relative = pos[0]-self.abs_pos[0]-self.anchor[0]*self.dim[0]
#         y_relative = -(pos[1]-self.abs_pos[1]-self.anchor[1]*self.dim[1])
#         x = self.projection.ax1_min_max[0] +  ((self.projection.ax1_min_max[1] - self.projection.ax1_min_max[0])*x_relative/(self.x_axis_length*self.dim[0]))
#         y = self.projection.ax2_min_max[0] +  ((self.projection.ax2_min_max[1] - self.projection.ax2_min_max[0])*y_relative/(self.y_axis_length*self.dim[1]))
#         return x, y
#
#     def draw(self, screen):
#         try:
#             pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)
#             if super(Scatterplot, self).draw(screen) and self.coordinates is not None:
#                 coord = self.coordinates
#                 if coord is not None:
#                     if coord.shape[0] > 1400:
#                         thickness = 2
#                     elif coord.shape[0] > 600:
#                         thickness = 3
#                     else:
#                         thickness = 4
#                     if self.is_labeled:
#                         Y = self.projection.Y
#                         if self.is_classification:
#                             Y_idxes = self.projection.Y_shifted
#                             if thickness == 2:
#                                 for i in range(0, len(Y)):
#                                     pygame.draw.line(screen, self.Y_colors[Y_idxes[i]], coord[i], coord[i], thickness)
#                             else:
#                                 for i in range(0, len(Y)):
#                                     pygame.draw.line(screen, self.Y_colors[Y_idxes[i]], coord[i], (coord[i,0], coord[i,1]+1), thickness)
#                                     pygame.draw.line(screen, self.Y_colors[Y_idxes[i]], (coord[i,0]-1, coord[i,1]), coord[i], thickness)
#                         else:
#                             Y_idxes = self.projection.Y_color_idxes
#                             for i in range(0, len(Y)):
#                                 pygame.draw.line(screen, self.Y_colors[Y_idxes[i]], coord[i], coord[i], thickness)
#                     else:
#                         for p in coord:
#                             pygame.draw.line(screen, self.color, p, p, thickness)
#                     for p in self.selected_points:
#                         pygame.draw.line(screen, (240, 230, 255), (coord[p][0],coord[p][1]-1), (coord[p][0],coord[p][1]+1), 2 )
#                     if len(self.draw_trajectory) > 2:
#                         grid_res, _ = self.draw_grid.shape
#                         prev_x = self.abs_pos[0]+self.dim[0]*self.anchor[0]+  self.dim[0]*self.x_axis_length*(self.draw_trajectory[0][0]/grid_res)
#                         prev_y = self.abs_pos[1]+self.dim[1]*(self.anchor[1]-self.y_axis_length) + self.y_axis_length*self.dim[1]*(self.draw_trajectory[0][1]/grid_res)
#                         for i in range(1, len(self.draw_trajectory)):
#                             now_x = self.abs_pos[0]+self.dim[0]*self.anchor[0]+  self.dim[0]*self.x_axis_length*(self.draw_trajectory[i][0]/grid_res)
#                             now_y = self.abs_pos[1]+self.dim[1]*(self.anchor[1]-self.y_axis_length) + self.y_axis_length*self.dim[1]*(self.draw_trajectory[i][1]/grid_res)
#                             pygame.draw.line(screen, (255, 102, 20), (prev_x,prev_y), (now_x,now_y), 2)
#                             prev_x, prev_y = now_x, now_y
#
#                 # mean indicators then stdev indicators
#                 pygame.draw.line(screen, self.color, (self.axis_mean_and_std_lines[0][0],self.axis_mean_and_std_lines[0][1]-3),\
#                              (self.axis_mean_and_std_lines[0][0],self.axis_mean_and_std_lines[0][1]+3), 2)
#                 pygame.draw.line(screen, self.color, (self.axis_mean_and_std_lines[1][0]-3,self.axis_mean_and_std_lines[1][1]),\
#                              (self.axis_mean_and_std_lines[1][0]+3,self.axis_mean_and_std_lines[1][1]), 2)
#                 pygame.draw.line(screen, self.color, (self.axis_mean_and_std_lines[0][0]-self.axis_mean_and_std_lines[2],self.axis_mean_and_std_lines[0][1]),\
#                              (self.axis_mean_and_std_lines[0][0]+self.axis_mean_and_std_lines[2],self.axis_mean_and_std_lines[0][1]), 3)
#                 pygame.draw.line(screen, self.color, (self.axis_mean_and_std_lines[1][0],self.axis_mean_and_std_lines[1][1]-self.axis_mean_and_std_lines[3]),\
#                              (self.axis_mean_and_std_lines[1][0],self.axis_mean_and_std_lines[1][1]+self.axis_mean_and_std_lines[3]), 3)
#             elif not self.converged and self.converging_view is not None:
#                 self.converging_view.draw(screen)
#             if self.selected:
#                 pygame.draw.rect(screen, self.color, (self.abs_pos, (self.x_axis_length*self.dim[0]+20,self.dim[1]+4)), 2)
#             if not self.close_button.hidden:
#                 self.close_button.draw(screen)
#             if (not self.drawing) and self.draw_heatmap:
#                 self.heatmap.draw(screen)
#         except: # if the scatterplot gets deleted and a model tries to draw in it
#             return
#
#     def set_projection(self, projection, is_labeled, is_classification, Y_colors):
#         1/0
#         self.projection = projection
#         self.rebuild_points()
#         self.is_labeled = is_labeled
#         self.is_classification = is_classification
#         self.Y_colors = Y_colors
#         self.can_be_selected = True
#
#     # compute the location of the points in pixel.
#     def rebuild_points(self):
#         if self.projection is None or self.projection.X is None: # not assigned a proj or proj not done yet
#             return
#         X = self.projection.X # setting these as local variables -> faster access than instance variables
#         ax_px_len = self.x_axis_length*self.dim[0]
#         ax1_min = self.projection.ax1_min_max[0]
#         ax2_min = self.projection.ax2_min_max[0]
#         ax1_wingspan = self.projection.ax1_min_max[1] - ax1_min
#         if ax1_wingspan == 0:
#             ax1_wingspan+=1e-12
#         ax2_wingspan = self.projection.ax2_min_max[1] - ax2_min
#         if ax2_wingspan == 0:
#             ax2_wingspan+=1e-12
#         x_offset = self.abs_pos[0] + self.anchor[0]*self.dim[0]
#         y_offset = self.abs_pos[1] + self.anchor[1]*self.dim[1]
#         locations = np.zeros(X.shape, dtype=int)
#         idx = 0
#         for obs in X:
#             locations[idx][0] = ax_px_len*((obs[0]-ax1_min)/ax1_wingspan)+x_offset
#             locations[idx][1] = y_offset-ax_px_len*((obs[1]-ax2_min)/ax2_wingspan)
#             idx += 1
#         self.coordinates = locations
#         ax1_mean = (ax_px_len*((self.projection.ax1_mean-ax1_min)/ax1_wingspan)+x_offset, y_offset)
#         ax2_mean = (x_offset, y_offset-ax_px_len*((self.projection.ax2_mean-ax2_min)/ax2_wingspan))
#         ax1_std  = ax_px_len*((self.projection.ax1_std)/ax1_wingspan)
#         ax2_std  = ax_px_len*((self.projection.ax2_std)/ax2_wingspan)
#         self.axis_mean_and_std_lines = [ax1_mean, ax2_mean, ax1_std, ax2_std]
#
#     def scale_to_square(self): # the parent container is likely not a square: we need to scale the axis to have equal lengths in pixel
#         axis_target_length = 0.88
#         yx_ratio = self.dim[1]/self.dim[0]
#         if yx_ratio < 1: # we need to separate both cases because we can only scale an axis by getting it smaller (or else we would have an axis bigger that the parent container)
#             self.x_axis_length = axis_target_length*yx_ratio
#             self.y_axis_length = axis_target_length
#         else:
#             self.x_axis_length = axis_target_length
#             self.y_axis_length = axis_target_length/yx_ratio
#         self.rebuild_axis()
#         self.rebuild_points()
#
#     def rebuild_axis(self):
#         self.lines.clear()
#         x_axis = [self.anchor, (self.anchor[0]+self.x_axis_length, self.anchor[1]), 1, self.color]
#         y_axis = [self.anchor, (self.anchor[0], self.anchor[1]-self.y_axis_length), 1, self.color]
#         small_bar_x = [(self.anchor[0]+self.x_axis_length, self.anchor[1]-0.01),(self.anchor[0]+self.x_axis_length, self.anchor[1]+0.01), 1, self.color]
#         small_bar_y = [(self.anchor[0]-0.01, self.anchor[1]-self.y_axis_length),(self.anchor[0]+0.01, self.anchor[1]-self.y_axis_length), 1, self.color]
#         self.add_line(x_axis)
#         self.add_line(y_axis)
#         self.add_line(small_bar_y)
#         self.add_line(small_bar_x)
#
#     def update_pos_and_dim(self):
#         self.dim     = (self.parent.dim[0]*self.dim_pct[0], self.parent.dim[1]*self.dim_pct[1])
#         self.rel_pos = (self.parent.dim[0]*self.pos_pct[0], self.parent.dim[1]*self.pos_pct[1])
#         self.abs_pos = (self.parent.abs_pos[0]+self.rel_pos[0], self.parent.abs_pos[1]+self.rel_pos[1])
#         self.background_rect = pygame.Rect(self.abs_pos, self.dim)
#         self.bounding_rect = pygame.Rect((self.abs_pos[0], self.abs_pos[1]), (self.dim[0], self.dim[1]+5))
#         self.scale_to_square()
#         self.rebuild_texts()
#         if self.converging_view is not None:
#             self.converging_view.update_pos_and_dim()
#         self.close_button.update_pos_and_dim()
#
#     def update_listener_booleans(self):
#         super(Scatterplot, self).update_listener_booleans()
#         if self.on_value_change is not None:
#             self.notify_on_value_change = True
#         if self.on_click_start is not None:
#             self.notify_on_click_start = True
#
#     def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
#         if not self.converged:
#             return
#         if not self.close_button.hidden and self.close_button.point_is_inside(mouse_pos):
#             self.close_button.propagate_hover(to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move)
#         elif self.notify_on_hover:
#             self.on_hover_listener.notify(mouse_pos, to_redraw)
#         return True # stop parent from propagating further
#
#     def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
#         if not self.converged:
#             return
#         if not self.selected:
#             self.on_selected_listener.notify(self.name, to_redraw)
#             return True
#         else:
#             if self.notify_on_click_start:
#                 self.on_click_start.notify((mouse_pos, mouse_button_status, pressed_special_keys), to_redraw)
#             if not self.close_button.hidden and self.close_button.point_is_inside(mouse_pos):
#                 self.close_button.propagate_mouse_down(to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release)
#             elif self.point_is_within_plot(mouse_pos):
#                 if mouse_button_status[0]:
#                     if not pressed_special_keys[0]:
#                         self.beeing_dragged = True
#                         self.drawing = False
#                         self.wipe_drawing()
#                         self.schedule_awaiting(awaiting_mouse_release)
#                         self.schedule_awaiting(awaiting_mouse_move)
#                     else:
#                         self.drawing = True
#                         self.schedule_awaiting(awaiting_mouse_release)
#                         self.schedule_awaiting(awaiting_mouse_move)
#                 elif mouse_button_status[1]:
#                     self.right_click_window.abs_pos = (mouse_pos[0]  , max(self.abs_pos[1], mouse_pos[1]-self.right_click_window.dim[1]))
#                     self.right_click_window.update_pos_and_dim()
#                     self.right_click_window.reset()
#                     windows.append(self.right_click_window)
#                     return True
#         return True
#
#     def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
#         if self.drawing:
#             self.close_drawing(release_pos)
#             self.schedule_draw(to_redraw)
#             self.on_click_listener.notify((release_pos, released_buttons, pressed_special_keys, True, False), to_redraw)
#         else:
#             if self.stop_awaiting_click:
#                 self.stop_awaiting_click = False
#             elif self.point_is_inside(release_pos):
#                 if self.notify_on_click:
#                     self.on_click_listener.notify((release_pos, released_buttons, pressed_special_keys, True, True), to_redraw)
#         self.beeing_dragged = False
#         self.drawing = False
#         return True
#
#     def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
#         if self.stop_awaiting_hover:
#             self.stop_awaiting_hover = False
#             return True
#         # check if still beeing clicked: else remove self from awaiting_mouse_move
#         if self.drawing:
#             self.add_drawing_point(mouse_positions[0], to_redraw)
#             return False
#         elif self.beeing_dragged:
#             self.on_drag_listener.notify((mouse_positions[0], mouse_button_status, pressed_special_keys, False, True), to_redraw)
#             return False
#         elif not self.close_button.hidden and pressed_special_keys[0]:
#             return False
#         return True
#
#     def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
#         print("awaited key press: ", pressed_keys)
#         return True















class Heatmap(Element):
    def __init__(self, pos_pct, dim_pct, name, parent, nb_variables, uid_generator, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None, on_value_change_listener=None):
        super(Heatmap, self).__init__(pos_pct, dim_pct, name, parent, uid=uid_generator.get(), on_Lclick_listener=on_Lclick_listener, on_Rclick_listener=on_Rclick_listener, on_hover_listener=on_hover_listener, on_scroll_listener=on_scroll_listener)
        self.nb_variables = nb_variables
        self.nb_row = int(np.ceil(np.sqrt(nb_variables)))
        self.squares_dim = 0.8*self.dim[1]/self.nb_row
        self.squares_colors   = np.zeros((self.nb_variables,), dtype=(int,3))
        self.squares_top_left = np.zeros((self.nb_variables,), dtype=(float,2))
        for i in range(nb_variables):
            self.squares_top_left[i] = (self.abs_pos[0] + self.squares_dim*(i%self.nb_row), self.abs_pos[1]+0.2*self.dim[1] + self.squares_dim*(int(i/self.nb_row)))
        self.label = Text((0.1, 0.04), 4, (1, 1), 16, "label: ", self, color=self.color)




    def delete(self):
        if self.label is not None:
            self.label.delete()
            self.label = None
        super(Heatmap, self).delete()

    def update_values(self, intensities, label):
        for i in range(self.nb_variables):
            self.squares_colors[i] = intensities[i]*self.color
        self.label.update("label: "+str(label), self.abs_pos)

    def update_pos_and_dim(self, mouse_pos):
        self.dim     = (self.parent.dim[0]*self.dim_pct[0], self.parent.dim[1]*self.dim_pct[1])
        self.squares_dim = 0.8*self.dim[1]/self.nb_row
        if mouse_pos[0] < self.parent.abs_pos[0]+self.parent.dim[0]/2:
            x = mouse_pos[0]+3
        else:
            x = mouse_pos[0]-self.dim[0]-3
        if mouse_pos[1] < self.parent.abs_pos[1]+self.parent.dim[1]/2:
            y = mouse_pos[1] + 1
        else:
            y = mouse_pos[1]-self.dim[1]-1
        self.abs_pos = (x, y)
        self.bounding_rect = pygame.Rect(self.abs_pos, (0.8*self.dim[1], self.dim[1]))
        for i in range(self.nb_variables):
            self.squares_top_left[i] = (self.abs_pos[0] + self.squares_dim*(i%self.nb_row), self.abs_pos[1]+0.2*self.dim[1] + self.squares_dim*(int(i/self.nb_row)))

    def attach_to_element(self, element, mouse_pos):
        self.parent = element
        self.update_pos_and_dim(mouse_pos)

    def draw(self, screen):
        if super(Heatmap, self).draw(screen):
            pygame.draw.rect(screen, self.background_color,self.bounding_rect)
            for s in range(self.squares_top_left.shape[0]):
                pygame.draw.rect(screen, self.squares_colors[s], pygame.Rect(self.squares_top_left[s],(self.squares_dim,self.squares_dim)))
            self.label.draw(screen)
