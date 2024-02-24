dataloader = torch.utils.data.DataLoader(portrait_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
