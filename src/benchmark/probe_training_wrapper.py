from .probe import ProbeTrainer

# train using embeddings
def train_embeddings(encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size, 
                 tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels, use_encoder=False, save_interval=100):

    if use_encoder:
      enc = encoder
    else:
      enc = None             
    
    probe_trainer = ProbeTrainer(encoder=enc,
                          epochs=num_epochs,
                          lr=lr,
                          batch_size=batch_size,
                          patience=patience,
                          wandb=wandb,
                          fully_supervised=False,
                          save_dir=save_dir,
                          representation_len=encoder.feature_size)
    probe_trainer.train(tr_episodes, val_episodes,
                      tr_labels, val_labels, save_interval=save_interval)

    final_accuracies, final_f1_scores = probe_trainer.test(test_episodes, test_labels)

    wandb.log(final_accuracies)
    wandb.log(final_f1_scores)

# train using images
def train_images(encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size,
                 tr_episodes, val_episodes, tr_labels, val_labels, 
                 test_episodes, test_labels, save_interval=100):
  
    probe_trainer = ProbeTrainer(encoder=encoder,
                          epochs=num_epochs,
                          lr=lr,
                          batch_size=batch_size,
                          patience=patience,
                          wandb=wandb,
                          fully_supervised=False,
                          save_dir=save_dir,
                          representation_len=encoder.feature_size)
    probe_trainer.train(tr_episodes, val_episodes,
                      tr_labels, val_labels, save_interval=save_interval)

    final_accuracies, final_f1_scores = probe_trainer.test(test_episodes, test_labels)

    wandb.log(final_accuracies)
    wandb.log(final_f1_scores)

# main training method
def run_probe_training(training_input, encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size, 
                 tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels, use_encoder=False, save_interval=100):
  
  if training_input == 'embeddings':
    train_embeddings(encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size,
                 tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels, use_encoder=use_encoder, save_interval=save_interval)
  elif training_input == 'images':
    train_images(encoder, probe_type, num_epochs, lr, patience, wandb, save_dir, batch_size,
                 tr_episodes, val_episodes, tr_labels, val_labels, 
                 test_episodes, test_labels, save_interval=save_interval)
  else:
    print("Invalid input...choose either 'embeddings' and 'images'")