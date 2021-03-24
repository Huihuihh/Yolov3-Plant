import sys
import json
import cv2
import os
import shutil

dataset = { "info": {
            "description": "Plant in COCO dataset.",
            "url": "",
            "version": "1.0",
            "year": "",
            "contributor": "tianhui",
            "date_created": "2021-03-16"},
            "images":[],
            "annotations":[],
            "categories": [
            {"supercategory:": "strawberry_healthy", "id": 1, "name": "strawberry_healthy"},
            {"supercategory:": "strawberry_Leaf_scorch", "id": 2, "name": "strawberry_Leaf_scorch"},
            {"supercategory:": "pepper_bacterial_spot", "id": 3, "name": "pepper_bacterial_spot"},
            {"supercategory:": "pepper_healthy", "id": 4, "name": "pepper_healthy"},
            {"supercategory:": "potato_eb", "id": 5, "name": "potato_eb"},
            {"supercategory:": "potato_lb", "id": 6, "name": "potato_lb"},
            {"supercategory:": "potato_healthy", "id": 7, "name": "potato_healthy"},
            {"supercategory:": "apple_scab", "id": 8, "name": "apple_scab"},
            {"supercategory:": "apple_black_rot", "id": 9, "name": "apple_black_rot"},
            {"supercategory:": "apple_healthy", "id": 10, "name": "apple_healthy"},
            {"supercategory:": "apple_cedar_rust", "id": 11, "name": "apple_cedar_rust"},
            {"supercategory:": "grape_black_rot", "id": 12, "name": "grape_black_rot"},
            {"supercategory:": "grape_blight", "id": 13, "name": "grape_blight"},
            {"supercategory:": "grape_esca", "id": 14, "name": "grape_esca"},
            {"supercategory:": "grape_healthy", "id": 15, "name": "grape_healthy"},
            {"supercategory:": "peach_healthy", "id": 16, "name": "peach_healthy"},
            {"supercategory:": "peach_bacterial_spot", "id": 17, "name": "peach_bacterial_spot"},
            {"supercategory:": "cherry_healthy", "id": 18, "name": "cherry_healthy"},
            {"supercategory:": "cherry_sour_powdery_mildew", "id": 19, "name": "cherry_sour_powdery_mildew"}
            ]
}

datapath = "data/images"
annopath = "data/labels"
trainsetfile = "data/plant.train.list"
outputpath = "plant-disease"
phase = "train"
classes = {"background": 0, "strawberry_healthy": 1, "strawberry_Leaf_scorch": 2, "pepper_bacterial_spot": 3, "pepper_healthy": 4, "potato_eb": 5, "potato_lb": 6, "potato_healthy": 7, "apple_scab": 8, "apple_black_rot": 9, "apple_healthy": 10, "apple_cedar_rust": 11, "grape_black_rot": 12, "grape_blight": 13, "grape_esca": 14, "grape_healthy": 15, "peach_healthy": 16, "peach_bacterial_spot": 17, "cherry_healthy": 18, "cherry_sour_powdery_mildew": 19}

with open(trainsetfile) as f:
    count = 1
    cnt = 0
    annoid = 0
    for line in f:
        cnt += 1
        line = line.strip()

        name,_ = os.path.basename(line).split('.')

        imagepath = os.path.join(datapath, name + ".jpg")
        # no obstacle currently drop it
        txtpath = os.path.join(annopath, name + ".txt")
        if not os.path.exists(txtpath):
            print(txtpath)
            continue

        im = cv2.imread(imagepath)

        height, width, channels = im.shape

        if cnt % 1000 == 0:
            print(cnt)

        dataset["images"].append({"license": 5, "file_name": line, "coco_url": "local", "height": height, "width": width, "flickr_url": "local", "id": cnt})
        with open(txtpath) as annof:
            annos = annof.readlines()

        for ii, anno in enumerate(annos):
            parts = anno.strip().split(' ')
            if len(parts) is not 5:
                continue
            class_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            if parts[0].find("group") == -1:
                iscrowd = 0
            else:
                iscrowd = 1

            annoid = annoid + 1

            class_id += 1 # start from 1 instead of 0
            x1 = int((x-w/2)*width)
            y1 = int((y-h/2)*height)
            wid = int(w*width)
            hei = int(h*height)

            dataset["annotations"].append({
                "segmentation": [],
                "iscrowd": iscrowd,
                "area": wid * hei,
                "image_id": cnt,
                "bbox": [x1, y1, wid, hei],
                "category_id": class_id,
                "id": annoid
            })
        count += 1

json_name = os.path.join(outputpath, "{}.json".format(phase))

with open(json_name, 'w') as f:
    json.dump(dataset, f)
