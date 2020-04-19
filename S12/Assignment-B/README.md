# Assignment-12 B

## #Annotation json file explanation

The main json object having all image annotation objects as child.

### Example of image annotation object

![Annotation Tree](https://github.com/rohitrnath/EVA-4/blob/master/S12/Assignment-B/annotationTree.png)

		{
		  "img_001.jpeg6029": {
		    "filename": "img_001.jpeg",
		    "size": 6029,
		    "regions": [
		      {
		        "shape_attributes": {
		          "name": "rect",
		          "x": 69,
		          "y": 36,
		          "width": 143,
		          "height": 122
		        },
		        "region_attributes": {
		          "name": "img_001",
		          "type": "Dog",
		          "image_quality": {
		            "good": true,
		            "frontal": true,
		            "good_illumination": true
		          }
		        }
		      }
		    ],
		    "file_attributes": {
		      "caption": "",
		      "public_domain": "no",
		      "image_url": ""
		    }
		  }


Each Image annotation has 4 immediate children elements. They are 1. filename 2. size 3. regions 4. file_attributes.

* filename -> Image file name

* size -> size of the image. 

* regions: This is the most important element for our object detection. These are the actual regions we draw around the object. 
		   Shape is the attribute of the region. Shape can be rectangle, polygon etc.
	A. rect -> shape is rectangle. x, y, width and height are the Shape attributes.
	B. shape _attributes:
		Position arguments defining the position of shape in the image. For VGG Image Annotator, position representing the top-left points of shape.
		x     ->  X co-ordinate of shape position.
		Y     ->  Y co-ordinate of shape position.
		width -> width of the rectangular region (x axis)
		height -> height of the rectangular region (y axis)

	C. region_attributes: We can configure the attributes. Possible to add/delete attributes.
		This covers the object type information.
		name -> name of the object (example img_001 we gave while annotating)
		type: type of the object e.g., human, cat, dog etc. Type we choose while annotating.
		image_quality: (We choose)
			frontal-> attribute tells whether image appears to at front of the image. 
			good_illumination-> true means, object is visible with good light.
			good -> true indicates, object quaility is good. Object can easily be identified.
			
4. file_attributes: All the attributes here are configurable. We can add and delete these attributes.
	Some of the default variable availagle inn VGG Image annotator is -
	caption -> caption to the image
	public_domain -> Wheather Image is subjected to public domain
	image_url: public url of the image
