from PIL import Image
from copy import copy
from os.path import join
from glob import glob
import functools

__all__ = ["pack_images"]

class Rect(object):
    """Represent a rectangle in the BinPack tree."""
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1  # bottom
        self.x2 = x2
        self.y2 = y2  # top

    def get_width(self):
        return abs(self.x2 - self.x1)

    def set_width(self, w):
        self.x2 = self.x1 + w

    def get_height(self):
        return abs(self.y2 - self.y1)

    def set_height(self, h):
        self.y2 = self.y1 + h

    def get_left(self):
        return self.x1

    def set_left(self, l):
        w = self.get_width()
        self.x1 = l
        self.x2 = l + w

    def get_top(self):
        return self.y2

    def set_top(self, t):
        h = self.get_height()
        self.y2 = t
        self.y1 = t - h

    def get_right(self):
        return self.x2

    def get_bottom(self):
        return self.y1

    def set_bottom(self, y1):
        h = self.get_height()
        self.y1 = y1
        self.y2 = self.y1 + h

    def offset(self, x, y):
        self.left = self.left + x
        self.top = self.top + y
        return self

    def inset(self, d):
        """return a rect which is this rect inset by d in each direction"""
        return Rect(self.x1 + d, self.y1 + d,
                    self.x2 - d, self.y2 - d)

    def inside(self, r):
        """return true if this rectangle is inside r"""
        return self.x1 >= r.x1 and self.x2 <= r.x2\
               and self.y1 >= r.y1 and self.y2 <= r.y2

    width = property(fget=get_width, fset=set_width)
    height = property(fget=get_height, fset=set_height)
    left = property(fget=get_left, fset=set_left)
    top = property(fget=get_top, fset=set_top)
    right = property(fget=get_right)
    bottom = property(fget=get_bottom, fset=set_bottom)

    def __str__(self):
        return "[%f, %f, %f, %f]" % (self.x1, self.y1, self.x2, self.y2)

    def __repr__(self):
        return "Rect[%s]" % str(self)

class BinPackNode(object):
    """A Node in a tree of recursively smaller areas within which images can be placed."""
    def __init__(self, area):
        """Create a binpack node
        @param area a Rect describing the area the node covers in texture coorinates
        """
        #the area that I take up in the image.
        self.area = area
        # if I've been subdivided then I always have a left/right child
        self.leftchild = None
        self.rightchild = None
        #I'm a leaf node and an image would be placed here, I can't be suddivided.
        self.filled = False

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, str(self.area))

    def insert(self, newarea):
        """Insert the newarea in to my area.
        @param newarea a Rect to insert in to me by subdividing myself up
        @return the area filled or None if the newarea couldn't be accommodated within this
            node tree
        """
        #if I've been subdivided already then get my child trees to insert the image.
        if self.leftchild and self.rightchild:
            return self.leftchild.insert(newarea) or self.rightchild.insert(newarea)

        #If my area has been used (filled) or the area requested is bigger then my
        # area return None. I can't help you.
        if self.filled or newarea.width > self.area.width or newarea.height > self.area.height:
            return None

        #if the image fits exactly in me then yep, the are has been filled
        if self.area.width == newarea.width and self.area.height == newarea.height:
            self.filled = True
            return self.area

        #I am going to subdivide myself, copy my area in to the two children
        # and then massage them to be useful sizes for placing the newarea.
        leftarea = copy(self.area)
        rightarea = copy(self.area)

        widthdifference = self.area.width - newarea.width
        heightdifference = self.area.height - newarea.height

        if widthdifference > heightdifference:
            leftarea.width = newarea.width
            rightarea.left = rightarea.left + newarea.width
            rightarea.width = rightarea.width - newarea.width
        else:
            leftarea.height = newarea.height
            rightarea.top = rightarea.top + newarea.height
            rightarea.height = rightarea.height - newarea.height

        #create my children and then insert it in to the left child which
        #was carefully crafted about to fit in one dimension.
        self.leftchild = BinPackNode(leftarea)
        self.rightchild = BinPackNode(rightarea)
        return self.leftchild.insert(newarea)

def _imagesize(i):
    return i.size[0] * i.size[1]

#table of heuristics to sort the list of images by before placing
# them in the BinPack Tree NOTE that they are compared backwards
# as we want to go from big to small (r2->r1 as opposed to r1->r2)
sort_heuristics = {
    "maxarea": lambda r1, r2:  cmp(_imagesize(r2[1]), _imagesize(r1[1])),
    "maxwidth": lambda r1, r2: cmp(r2[1].size[0], r1[1].size[0]),
    "maxheight": lambda r1, r2: cmp(r2[1].size[1], r1[1].size[1]),
}

def pack_images(imagelist, dstfilename, padding=2, sort="maxarea"):
    """pack the images in image list in to a pow2 PNg file
    @param imagelist iterable of tuples (image name, image)
    @param padding padding to be applied to all sides of the image
    @param dstfilename the filename to save the packed image to.
    @return a list of ( rect, name, image) tuples describing where the images were placed.
    """

    #sort the images based on the heuristic passed in
    images = sorted(imagelist, key=functools.cmp_to_key(sort_heuristics[sort]))

    #the start dimension of the target image. this grows
    # by doubling to accomodate the images. Should start
    # on a power of two otherwise it wont end on a power
    # of two. Could possibly start this on the first pow2
    # above the largest image but this works.
    targetdim_x = 64
    targetdim_y = 64
    placement = []
    while True:
        try:
            placement = []
            tree = BinPackNode(Rect(0, 0, targetdim_x, targetdim_y))

            #insert each image into the BinPackNode area. If an image fails to insert
            # we start again with a slightly bigger target size.
            for name, img in images:
                imsize = img.size
                r = Rect(0, 0, imsize[0] + padding * 2, imsize[1] + padding * 2)
                uv = tree.insert(r)
                if uv is None:
                    #the tree couldn't accomodate the area, we'll need to start again.
                    raise ValueError('Pack size too small.')
                uv = uv.inset(padding)
                placement.append((uv, name, img))

            #if we get here we've found a place for all the images so
            # break from the while True loop
            break
        except ValueError:
            if targetdim_x == targetdim_y:
                targetdim_x = targetdim_x * 2
            else:
                targetdim_y = targetdim_x

    # save the images to the target file packed
    image = Image.new("RGBA", (targetdim_x, targetdim_y))
    for uv, name, img in placement:
        image.paste(img, (uv.x1, uv.y1))

    # image.show()
    image.save(dstfilename)

    return placement
