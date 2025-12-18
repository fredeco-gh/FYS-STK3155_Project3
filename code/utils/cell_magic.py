from IPython.core.magic import register_cell_magic
from IPython.core.getipython import get_ipython

@register_cell_magic
def skip_if(line, cell):
    """Jupyter cell magic to skip execution of a cell based on a condition."""
    global_scope = get_ipython().user_global_ns
    if eval(line, global_scope):
        return  # Skip execution if the condition is True
    get_ipython().run_cell(cell) # Execute the cell if the condition is False