# Vector representation of molecules
import numpy as np


def add_vector_rep(view, r, e, colors=None):
    """ Add vectors to a NGLViewer.

    Parameters
    ==========
    view : NGLViewer object
        The NGL view object to add the arrow to.
    r : array (N,3)
        Numpy array of positions.
    e : array (N,3)
        Numpy array of directions.
    colors : list
        List of RGB colors
    """

    # We create the arrows by using a shape buffer as this puts the least
    # pressure on the gpu. To do this we need a list of start and end points of
    # our arrows.
    n = np.shape(r)[0] * np.shape(r)[1]
    start = np.reshape(r - e / 2, [n,])
    end = np.reshape(r + e / 2, [n,])

    # The radii of the arrows are simply set to 1.
    radii = np.array([1.0] * n)

    if colors == None:
        # Default to blue ([0,0,10]) arrows
        colors = np.reshape(np.array([[0, 0, 10],] * np.shape(r)[0]), [n,])
        view.shape.add_buffer(
            "arrow",
            position1=start.tolist(),
            position2=end.tolist(),
            color=colors.tolist(),
            radius=radii.tolist(),
        )
    else:
        view.shape.add_buffer(
            "arrow",
            position1=start.tolist(),
            position2=end.tolist(),
            color=colors.tolist(),
            radius=radii.tolist(),
        )

    # Make sure we add a representation corresponding to the arrows for proper
    # indexing.
    view.add_representation("buffer")
