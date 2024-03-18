import sys, os
from utils import get_gui_config, luminosity_change
from engine.gui.shared_variable import Shared_variable
from engine.gui.gui import Gui, Id_generator
from engine.gui.container import Container
from engine.gui.window import Window
from engine.screen_managers.main_manager import Main_manager
from engine.screen_managers.absQA_manager import absQA_manager
from engine.screen_managers.relQA_manager import relQA_manager
import pygame
import numpy as np



def make_main_screen(theme, config, uid_generator, absQA_manager, relQA_manager):
	main_window = Window((0,0), config["resolution"], "main window", close_on_click_outside=False, uid_generator=uid_generator, color=theme["main"]["color"], background_color=theme["main"]["background"])
	manager     = Main_manager(config, main_window, theme["main"], uid_generator, absQA_manager=absQA_manager, relQA_manager=relQA_manager)
	return {"manager":manager, "window":main_window}

def make_abs_QA_screen(theme, config, uid_generator):
	window  = Window((0,0), config["resolution"], "abs QA window", close_on_click_outside=False, uid_generator=uid_generator, color=theme["absQA"]["color"], background_color=theme["main"]["background"])
	manager = absQA_manager(window, theme["absQA"], uid_generator)
	return {"manager":manager, "window":window}

def make_rel_QA_screen(theme, config, uid_generator):
	window  = Window((0,0), config["resolution"], "rel QA window", close_on_click_outside=False, uid_generator=uid_generator, color=theme["relQA"]["color"], background_color=theme["main"]["background"])
	manager = relQA_manager(window, theme["relQA"], uid_generator)
	return {"manager":manager, "window":window}

def make_theme(print_mode):
	main_theme  = {"background" : np.array([5,0,10]), "color" : np.array([0, 200,250])}
	relQA_theme = {"background" : np.array([0, 6, 2]), "color" : np.array([15, 222, 236])}
	absQA_theme = {"background" : np.array([6, 4, 1]), "color" : np.array([255, 122, 36])}
	if print_mode:
		main_theme["background"]  = np.array([255,255,255])
		relQA_theme["background"] = np.array([255,255,255])
		absQA_theme["background"] = np.array([255,255,255])
		main_theme["color"]  = luminosity_change(main_theme["color"], -300)
		relQA_theme["color"] = luminosity_change(relQA_theme["color"], -300)
		absQA_theme["color"] = luminosity_change(absQA_theme["color"], -300)
	return {"main" : main_theme,"relQA" : relQA_theme, "absQA" : absQA_theme}

def run(argv):
	uid_generator = Id_generator()
	config        = get_gui_config(argv)
	display       = init_screen(config)
	running_flag  = Shared_variable(True)
	theme		  = make_theme(config["print mode"])
	absQA_screen  = make_abs_QA_screen(theme, config, uid_generator)
	relQA_screen  = make_rel_QA_screen(theme, config, uid_generator)
	main_screen   = make_main_screen(theme, config, uid_generator, absQA_manager=absQA_screen["manager"], relQA_manager=relQA_screen["manager"])
	absQA_screen["manager"].attach_main_manager(main_screen["manager"])
	relQA_screen["manager"].attach_main_manager(main_screen["manager"])
	gui = Gui(display, theme, config, running_flag, main_screen, absQA_screen, relQA_screen)
	gui.routine()

def init_screen(config):
	if not config["windowed"]:
		pygame.display.init()
		screen_info = pygame.display.Info()
		config["resolution"] = (screen_info.current_w, screen_info.current_h)
		return pygame.display.set_mode(config["resolution"], pygame.FULLSCREEN)
	else:
		return pygame.display.set_mode(config["resolution"])



if __name__ == "__main__":
    run(sys.argv[1:])
