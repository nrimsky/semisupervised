"""
Model(
  (body): Body(
    (conv): Sequential(
      (0): ConvBodyBlock(
        (block): Sequential(
          (0): Conv2d(12, 8, kernel_size=(4, 1), stride=(1, 1))
          (1): ReLU()
          (2): AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)
        )
      )
      (1): ConvBodyBlock(
        (block): Sequential(
          (0): Conv2d(8, 6, kernel_size=(4, 1), stride=(1, 1))
          (1): ReLU()
          (2): AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)
        )
      )
    )
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (head): ClassificationHead(
    (dropout): Dropout(p=0.1, inplace=False)
    (linear): Linear(in_features=42, out_features=4, bias=True)
    (cos_sim): CosineSimilarity()
  )
)
"""

