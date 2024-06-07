model_path="download/inference/pretrained_checkpoints/TCGA/omics_diffusion_model406000.pt"
classifier_path="download/inference/pretrained_checkpoints/TCGA/classifier_model149999.pt"
class_mapping="TCGA-BLCA:0,TCGA-BRCA:1,TCGA-COAD:2,TCGA-ESCA:3,TCGA-HNSC:4,TCGA-KICH:5,TCGA-KIRC:6,TCGA-KIRP:7,TCGA-LIHC:8,TCGA-LUAD:9,TCGA-LUSC:10,TCGA-OV:11,TCGA-PAAD:12,TCGA-PRAD:13,TCGA-READ:14,TCGA-SARC:15,TCGA-STAD:16,TCGA-THCA:17,TCGA-UCEC:18"
model_type="omics_unet"
genomics_table_fp="download/inference/demo_data/TCGA-general/genomics_table.csv"
transcriptomics_table_fp="download/inference/demo_data/TCGA-general/transcriptomics_table.csv"

DATA_OPTIONS="--data_dir $DATA_DIR \
--class_mapping $class_mapping \
--genomics_table_fp $genomics_table_fp \
--transcriptomics_table_fp $transcriptomics_table_fp \
"