# -*- coding: utf-8 -*-
import cStringIO
import os
import numpy as np
import PIL
from PIL.Image import fromarray
from PIL import Image as PILImage
from shapely.geometry import Polygon, box
from sldc import TileExtractionException, alpha_rasterize, Image, Tile, TileBuilder

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def _get_crop(cytomine, image_inst, geometry):
    """
    Download the crop corresponding to bounds on the given image instance
    from cytomine

    Parameters
    ----------
    cytomine : :class:`Cytomine`
        The cilent holding the communication
    image_inst : :class:`ImageInstance` or image instance id (int)
        The image on which to extract crop
    geometry: tuple (int, int, int, int)
        The information about the geometry of the crop structured as (offset_x, offset_y, width, height)
    zoom: int (optional, default=0)
        The zoom to apply to the image
    """
    bounds = dict()
    bounds["x"], bounds["y"], bounds["w"], bounds["h"] = geometry
    url = "{}{}{}{}".format(cytomine._Cytomine__protocol, cytomine._Cytomine__host, cytomine._Cytomine__base_path,
                            image_inst.get_crop_url(bounds))
    resp, content = cytomine.fetch_url(url)
    if resp.status != 200:
        raise IOError("Couldn't fetch the crop for image {} and bounds {} from server at url {} (status : {}).".format(image_inst.id, geometry, url, resp.status))
    tmp = cStringIO.StringIO(content)
    pil_image = PILImage.open(tmp)  # fetch the image but this process inverts R and B channels (?)
    return np.asarray(pil_image)[:, :, (2, 1, 0)]  # so reorder to channel to have a valid RGB image


class CytomineSlide(Image):
    """
    A slide from a cytomine project
    """

    def __init__(self, cytomine, id_img_instance):
        """Construct CytomineSlide objects

        Parameters
        ----------
        cytomine: cytomine.Cytomine
            The cytomine client
        id_img_instance: int
            The id of the image instance
        """
        self._cytomine = cytomine
        self._img_instance = self._cytomine.get_image_instance(id_img_instance, include_server_urls=True)

    @property
    def image_instance(self):
        return self._img_instance

    @property
    def cytomine(self):
        return self._cytomine

    @property
    def np_image(self):
        raise NotImplementedError("Disabled due to the too heavy size of the images")

    @property
    def width(self):
        return self._img_instance.width

    @property
    def height(self):
        return self._img_instance.height

    @property
    def channels(self):
        return 3

    def __getstate__(self):
        self._cytomine.__conn = None
        return self.__dict__

    def __str__(self):
        return "CytomineSlide (#{}) ({} x {})".format(self._img_instance.id, self.width, self.height)


class CytomineTile(Tile):
    """
    A tile from a cytomine slide
    """
    def __init__(self, cytomine, working_path, parent, offset, width, height, tile_identifier=None, polygon_mask=None):
        """Constructor for CytomineTile objects

        Parameters
        ----------
        cytomine: cytomine.Cytomine
            An initialized instance of the cytomine client
        parent: Image
            The image from which is extracted the tile
        offset: (int, int)
            The x and y coordinates of the pixel at the origin point of the slide in the parent image.
            Coordinates order is the following : (x, y).
        width: int
            The width of the tile
        height: int
            The height of the tile
        tile_identifier: int, optional (default: None)
            A integer identifier that identifies uniquely the tile among a set of tiles
        polygon_mask: Polygon (optional, default: None)
            The polygon representing the alpha mask to apply to the tile window

        Notes
        -----
        The coordinates origin is the leftmost pixel at the top of the slide
        """
        Tile.__init__(self, parent, offset, width, height, tile_identifier=tile_identifier, polygon_mask=polygon_mask)
        self._cytomine = cytomine
        self._working_path = working_path

    @property
    def np_image(self):
        try:
            image_instance = self.base_image.image_instance
            x, y, width, height = self.abs_offset_x, self.abs_offset_y, self.width, self.height

            # check if the tile was cached
            cache_file = "{}_{}_{}_{}_{}.png".format(image_instance.id, x, y, width, height)
            cache_path = os.path.join(self._working_path, cache_file)
            if os.path.exists(cache_path):
                return np.asarray(PIL.Image.open(cache_path))

            # build crop box
            tbox = (x, y, width, height)
            # fetch image
            np_array = np.asarray(_get_crop(self._cytomine, image_instance, tbox))
            if np_array.shape[1] != tbox[2] or np_array.shape[0] != tbox[3] \
                    or np_array.shape[2] < self._underlying_image_channels:
                msg = "Fetched image has invalid size : {} instead of {}".format(np_array.shape, (tbox[3], tbox[2], self.channels))
                raise TileExtractionException(msg)
            # drop alpha channel if there is one
            if np_array.shape[2] >= 4:
                np_array = np_array[:, :, 0:3]
            np_array = np_array.astype("uint8")
            if self.polygon_mask is None:
                return np_array
            else:
                try:
                    return alpha_rasterize(np_array, self.polygon_mask)
                except ValueError:
                    return np_array
        except IOError as e:
            raise TileExtractionException(e.message)

    @property
    def channels(self):
        return 3 if self.polygon_mask is None else 4

    def _tile_box(self):
        offset_x, offset_y = self.abs_offset
        return box(offset_x, offset_y, offset_x + self.width, offset_y + self.height)

    def __getstate__(self):
        self._cytomine._Cytomine__conn = None  # delete socket to make the tile serializable
        return self.__dict__


class CytomineTileBuilder(TileBuilder):
    """
    A builder for CytomineTile objects
    """

    def __init__(self, cytomine, working_path):
        """Construct CytomineTileBuilder objects

        Parameters
        ----------
        cytomine: cytomine.Cytomine
            The initialized cytomine client
        """
        self._cytomine = cytomine
        self._working_path = working_path

    def build(self, image, offset, width, height, polygon_mask=None):
        return CytomineTile(self._cytomine, self._working_path, image, offset, width, height, polygon_mask=polygon_mask)


class TileCache(object):
    """A class to use jointly with tiles to avoid fetching them several time
    """
    def __init__(self, tile_builder, working_path):
        self._tile_builder = tile_builder
        self._working_path = working_path

    def fetch_and_cache(self, tile):
        """Fetch the np_image for the passed tile and cache it in the working path. If the np_image was already
        cached nothing is fetched from the server.
        Parameters
        ----------
        tile: Tile
            The tile of which the np_image must be fetched and cached
        Returns
        -------
        path: string
            The full path to which was cached the np_image
        """
        if not self._cache_has(tile, alpha=False):
            self._save(tile, tile.np_image.astype("uint8"), alpha=False)
        return self._tile_path(tile, alpha=False)

    def polygon_fetch_and_cache(self, image, polygon, alpha=True):
        """Fetch the np_image for the tile boxing the passed polygon and cache it in the working path. If the np_image
        was already cached nothing is fetched from the server.
        Parameters
        ----------
        image: Image
            The image from which the tile must be extracted
        polygon: Polygon
            The polygon that should be boxed by the tile
        alpha: bool
            True of applying an alpha mask structured like the polygon

        Returns
        -------
        path: string
            The full path to which was cached the np_image
        """
        tile = image.tile_from_polygon(self._tile_builder, polygon, mask=alpha)
        if not self._cache_has(tile, alpha=alpha):
            np_image = tile.np_image
            self._save(tile, np_image.astype("uint8"), alpha=alpha)
        return self._tile_path(tile, alpha)

    def tile_np_image(self, tile):
        """Get the np_image of the given tile from the cache. If it was not cached, fetch it from the
        server and cache it before returning it.

        Parameters
        ----------
        tile: Tile
            The tile of which the np_image must be fetched and cached

        Returns
        -------
        np_image: array-like
            The image representation
        """
        path = self.fetch_and_cache(tile)
        return np.asarray(PIL.Image.open(path)).astype("uint8")

    def polygon_np_image(self, image, polygon, alpha=True):
        """Get the np_image of the tile that boxes the polygon in the image from the cache. If it was not cached,
        fetch it from the server and cache it before returning it.

        Parameters
        ----------
        image: Image
            The image from which the tile must be extracted
        polygon: Polygon
            The polygon that should be boxed by the tile
        alpha: bool
            True of applying an alpha mask structured like the polygon

        Returns
        -------
        np_image: array-like
            The image representation
        """
        return self.tile_np_image(image.tile_from_polygon(self._tile_builder, polygon, mask=alpha))

    def _save(self, tile, np_image, alpha=False):
        """Save the tile np_image at the path produced by _tile_path
        Parameters
        ----------
        tile: Tile
            The tile from which was generated np_image
        np_image: array-like
            The numpy image to save
        alpha: bool (optional, default: False)
            True if the np_image has an alpha channel
        """
        fromarray(np_image).save(self._tile_path(tile, alpha))

    def _cache_has(self, tile, alpha=False):
        """Check whether the given tile was already cached by the tile cache
        Parameters
        ----------
        tile: Tile
            The tile
        alpha: bool (optional, default: False)
            True if the alp
        :return:
        """
        return os.path.isfile(self._tile_path(tile, alpha))

    def _tile_path(self, tile, alpha=False):
        """Return the path where to store the tile

        Parameters
        ----------
        tile: Tile
            The tile object containing the image to store
        alpha: bool (optional, default: False)
            True if an alpha mask is applied

        Returns
        -------
        path: string
            The path in which to store the image
        """
        basename = "{}_{}_{}_{}_{}".format(tile.base_image.image_instance.id, tile.offset_x,
                                           tile.offset_y, tile.width, tile.height)
        if alpha:
            basename = "{}_alpha".format(basename)
        return os.path.join(self._working_path, "{}.png".format(basename))
