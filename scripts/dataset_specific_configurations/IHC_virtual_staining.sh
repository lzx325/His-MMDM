model_path="download/inference/pretrained_checkpoints/IHC_virtual_staining/diffusion_model308000.pt"
classifier_path="download/inference/pretrained_checkpoints/IHC_virtual_staining/classifier_model149999.pt"
model_type="unet"

DATA_OPTIONS="--data_dir $DATA_DIR \
--class_mapping ACTH:0,ATRX:1,CD34:2,CD56:3,CK:4,CgA:5,EMA:6,FSH:7,GFAP:8,GH:9,HE:10,IDH-1:11,Ki67:12,LH:13,MGMT:14,NF:15,Neu-N:16,Oligo-2:17,P53:18,PR:19,PRL:20,S-100:21,SSTR2:22,Syn:23,TSH:24,Vimentin:25 \
"