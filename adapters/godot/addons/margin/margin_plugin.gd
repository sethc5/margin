@tool
extends EditorPlugin


func _enter_tree() -> void:
	add_autoload_singleton("Margin", "res://addons/margin/margin_autoload.gd")


func _exit_tree() -> void:
	remove_autoload_singleton("Margin")
