# Launch the training script using the 'accelerate' library
accelerate launch diffusers/examples/research_projects/multi_subject_dreambooth \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \  # Specify the pretrained model
  --concepts_list='path/to/concept_list.json' \  # Specify the concepts list
  --output_dir='path/to/output' \  # Specify the output directory
  --train_text_encoder \  # Enable training of the text encoder
  --resolution=512 \  # Set the resolution of the images
  --num_train_epochs=2 \  # Set the number of training epochs
  --train_batch_size=1 \  # Set the training batch size
  --gradient_checkpointing \  # Enable gradient checkpointing
  --learning_rate=5e-6 \  # Set the learning rate
  --lr_scheduler='constant' \  # Set the learning rate scheduler to 'constant'
  --report_to='wandb' \  # Report training progress to Weights & Biases
  --lr_warmup_steps=0 \  # Set the number of learning rate warmup steps
  --checkpointing_steps=2500 \  # Set the checkpointing steps
  --lr_scheduler='cosine' \  # Set the learning rate scheduler to 'cosine'
  --mixed_precision="fp16" \  # Enable mixed precision training
  --validation_prompt='A HoPla Sugarbeet' \  # Set the validation prompt
  --max_train_steps=60000  # Set the maximum number of training steps
