
python merge.py -i \
    ~/data/coco_rec_multi_1118/rec_multi_1118 \
    ~/data/Renewsys-Unsolder/Renewsys-Unsolder \
    ~/data/Renewsys-Cracks/Renewsys-Cracks \
    -a \
    ~/data/coco_rec_multi_1118/annotations/instances_rec_multi_1118.json \
    ~/data/Renewsys-Unsolder/annotations/Renewsys-Unsolder.json \
    ~/data/Renewsys-Cracks/Renewsys-Cracks.json \
    -o \
    ~/data/defect-rec-multi

