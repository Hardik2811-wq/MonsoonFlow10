I'll write a split script for the 2005–2018 / 2019–2020 / 2021 plan and test it on a synthetic file shaped like the official data.

I wrote split_data.py, which splits the official dataset into three parts by year, oldest first:

Part	Years	Days	10-day windows	What it's for
Train	2005–2018	5,113	5,101	Fitting the model, plus the normalisation stats and climatology
Val	2019–2020	731	719	Choosing a model, early stopping, tuning
Test	2021	365	353	Kept sealed; run once, after the model is final

I couldn't download the real india_climate_train.nc here. I tested the script on a fake file with the same dates, grid and channels, in both likely layouts: one variable per channel, or a single variable with a channel dimension. It may still need a small tweak once it meets the real file. It prints the variables it finds, so any mismatch should be easy to spot.

Run it with:

bash
python split_data.py --input india_climate_train.nc --out data/splits

What it saves:

train.npz, val.npz and test.npz: the weather data in shape [days, 6, 32, 36], plus dates and day-of-year.
train_stats.json and train_climatology.npy: built from the training years only, so the validation and test years can't leak in. Your team should normalise with these in both training and serve.py, since the kit's own stats.json may include all years.
static.npy (land-sea mask and terrain height) and splits.json (a summary of the split).

Choices you can change:

Training windows are built inside each part, so a target day never crosses into another part. --gap-days 7 also drops the first week of val and test for extra separation.
The year ranges can be changed with --train-years, --val-years and --test-years.
Keep the .nc and .npz files out of the submission repo, because the organisers' repo scanner flags shipped weather data.