import numpy as np
import matplotlib.pyplot as plt 
from cv2 import kmeans, KMEANS_RANDOM_CENTERS, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER
from PIL import Image
from io import BytesIO


class Picture:
    """A picture object
    Inputs image and makes a new image with color data on it.
    
    How it works:
        1. Posterized image (with decreased colors range) fed to KMeans alghoritm
        2. from KMeans we receiving most interesting colors
        3. from this array taking k-colors and making a histogram

    Args:
        image_b (bytes): image in bytesIO representation
        k (int): count of colors
    
    Constants:
        PAD (bool): if True, makes white border on image, otherwise nothing
        MIN_H (int): minimal image height
        MIN_W (int): minimal image width

    Attributes:
        image (numpy.ndarray): sourse image
        image_posterized (numpy.ndarray): image with quantizated colors (decreased colors range)
        image_pil (PIL.Image): image in PIL representation
        label (numpy.ndarray): Picture.center counts
        center (numpy.ndarray): most wanted colors in RGB representation
        hist (numpy.ndarray): histogram of k most wanted colors
    
    """
    PAD = True
    MIN_W = 1000
    MIN_H = 1000

    def __init__(self, image_b, k=6):
        self.image_pil = Image.open(BytesIO(image_b))
        self.image = np.array(self.image_pil)
        self.k = k

        #if image is less than minimal width and height
        if (self.image.shape[0] < self.MIN_W) or (self.image.shape[1] < self.MIN_H):

            #choose maximal proportion (where side less than minimal)
            p = max(self.MIN_W / self.image.shape[0], 
                    self.MIN_H / self.image.shape[1])

            #increase both sides by proportional multiplication
            self.image_pil = self.image_pil.resize((int(p*self.image.shape[1]), int(p*self.image.shape[0])), Image.LANCZOS)
            self.image = np.array(self.image_pil)

        #quantization - decrease colors range
        self.image_posterized = self.image_pil.quantize((self.k//2)*self.k**2, 1)
        self.image_posterized = self.image_posterized.convert("RGB", palette=Image.ADAPTIVE, colors=self.k**3)
        self.image_posterized = np.array(self.image_posterized)

        """
        KMeans:
            1. create random centers of colors (centroids)
            2. for each color assign nearest centroid (calculates as Euclidian distance)
            3. move centroid to their new position (mean of assigned colors)
            4. repeat stepts 2 and 3 until sum of distances is minimal
        """
        _, self.label, self.center = kmeans(
                    self.image_posterized.reshape(-1, 3).astype('float32'), 
                    self.k, 
                    None, 
                    (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 10, 1), 
                    0,
                    KMEANS_RANDOM_CENTERS
                    )

        self.hist = self.get_hist()

        if self.PAD:
            pv = max(self.image_pil.width, self.image_pil.height)//85
            self.image_pil = Image.fromarray(np.pad(self.image_pil, ((pv,pv), (pv,pv), (0,0)), 'constant', constant_values=255))
        
    def get_hist(self):
        (hist, _) = np.histogram(self.label, bins=self.k)
        mask = np.argsort(hist)
        self.center = self.center.reshape(-1, 3)[mask]
        return hist[mask]

    #picture to the left of the form
    def __add__(self, form):
        assert isinstance(form, Form), 'Should be From object'
        return form.__radd__(self)
    
    #picture over the form
    def __radd__(self, form):
        assert isinstance(form, Form), 'Should be From object'
        return form.__add__(self)


class Form:
    """A form object
    Only for work with Picture

    Args:
        flag (bool): for ultra wide (0) or ultra tall (1)
        **params:
            w,h, r,c, an1,an2, bb, loc - better you don't touch this, because there is no meaning :)

    """
    def __init__(self, flag, **params):
        self.flag = flag
        self.params = params

    def create_form(self, image, colors, hist):
        """Create form and insert image into it
        
        Args:
            image (Image): image to insert into form
            colors (numpy.ndarray): Picture.center
            hist (numpy.ndarray): Picture.hist

        Returns:
            PIL.Image - form with a picture

        """
        fig = plt.figure()
        fig.set_size_inches(self.params['w'], self.params['h'])

        ax1 = plt.subplot(self.params['r'], self.params['c'], 2, aspect="equal", anchor=self.params['an2'])
        ax2 = plt.subplot(self.params['r'], self.params['c'], 1, aspect="equal", anchor=self.params['an1'])

        hex_colors = np.array(['#{:02X}{:02X}{:02X}'.format(x[0],x[1],x[2]) for x in colors.astype('uint8')])
        wedges, _ = ax1.pie(hist, colors=hex_colors, startangle=90, radius=1.25)
        ax1.legend(wedges, hex_colors, loc=self.params['loc'], bbox_to_anchor=self.params['bb'], fontsize=90+self.flag*20, 
                    labelspacing=0.75 + 0.25*self.flag*((len(hist)-hist.size) / (len(hist)*2)))

        ax2.imshow(image)
        ax2.axis('off')
        
        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(w, h, 3)
        plt.close()

        return Image.frombytes("RGB", (w, h), buf.tostring())

    #picture to the left of the form
    def __radd__(self, pic):
        assert isinstance(pic, Picture), 'Should be Picture object'

        self.figure = self.create_form(
            pic.image_posterized, 
            pic.center, 
            pic.hist
            )

        w1, h1 = pic.image_pil.width, pic.image_pil.height
        w2, h2 = self.figure.width, self.figure.height

        if h1<h2: self.figure = self.figure.resize((int(w2*(h1/h2)), h1), Image.LANCZOS)
        if h1>h2: pic.image_pil = pic.image_pil.resize((int(w1*(h2/h1)), h2), Image.LANCZOS)

        return Image.fromarray(np.hstack((np.asarray(pic.image_pil), np.asarray(self.figure))))
    
    #picture over the form
    def __add__(self, pic):
        assert isinstance(pic, Picture), 'Should be Picture object'

        self.figure = self.create_form(
            pic.image_posterized, 
            pic.center, 
            pic.hist
            )
        
        w1, h1 = pic.image_pil.width, pic.image_pil.height
        w2, h2 = self.figure.width, self.figure.height

        if w1<w2: self.figure = self.figure.resize((w1, int(h2*(w1/w2))), Image.LANCZOS)
        if w1>w2: pic.image_pil = pic.image_pil.resize((w2, int(h1*(w2/w1))), Image.LANCZOS)

        return Image.fromarray(np.vstack((np.asarray(pic.image_pil), np.asarray(self.figure))))

def vegnn(imgb):
    """Mediator between Form and Picture

    Args:
        imgb (bytes): image in bytes

    Returns:
        result - image in bytes
    
    """
    image = Picture(imgb)
    result = None

    if (image.image_pil.width-image.image_pil.height) > (image.image_pil.width+image.image_pil.height)/10: 
        form = Form(flag=0, w=50, h=15, r=1, c=2, an1='C', an2='W', bb=(1,0,-0.5,1), loc="center left")
        ans = form + image
    else: 
        form = Form(flag=1, w=15, h=50, r=2, c=1, an1='S', an2='N', bb=(0.5,1), loc='lower center')
        ans = image + form

    with BytesIO() as output:
        ans.save(output, 'BMP')
        result = output.getvalue()

    return result


if __name__ == '__main__':
    import requests
    EXTRA_TEST = 'https://pp.userapi.com/c854328/v854328840/24b3a/zB5HXYFdz4I.jpg'
    test = vegnn(requests.get(EXTRA_TEST).content)
    plt.imshow(np.array(Image.open(BytesIO(test.result))))
    plt.show()