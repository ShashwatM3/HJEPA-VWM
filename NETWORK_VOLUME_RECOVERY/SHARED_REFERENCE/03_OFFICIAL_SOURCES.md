# Official and primary sources

Last checked: **2026-07-23 (Asia/Kolkata)**. Links point directly to the pages supporting the
runbook. Dynamic pricing and console UI can change; recheck before a later restore.

## RunPod

1. [RunPod S3-compatible API](https://docs.runpod.io/storage/s3-api)
   - direct network-volume access without launching a Pod;
   - using the S3-compatible API does not change network-volume pricing;
   - separate S3 API key, one-time secret, endpoint/region configuration;
   - `/workspace/x` to `s3://VOLUME_ID/x` path mapping;
   - US-MO-1 endpoint;
   - warning that `sync`/listing can be slow or fail above 10,000 files or 10 GB;
   - first-list ETag/MD5 computation, pagination retry guidance, multipart and 4 TB file limit;
   - exact supported/unsupported operations, including no `DeleteObjects`, bucket location,
     versioning, tags, ACLs, bucket policies, encryption controls, object lock, or presigned URLs.
2. [RunPod network volumes](https://docs.runpod.io/storage/network-volumes)
   - standard storage pricing: $0.07/GB-month first 1 TB and $0.05 beyond;
   - data persists independently from Pods but can be terminated when unpaid;
   - terminated volume data cannot be recovered;
   - volume creation, supported size growth/no shrink, attach-at-deploy behavior, `/workspace` mount.
3. [RunPod Pod pricing and storage](https://docs.runpod.io/pods/pricing)
   - per-second compute, no RunPod Pod data ingress/egress fees;
   - storage billing and explicit statement that RunPod is not designed for long-term cloud storage;
   - one-hour-credit deployment minimum and stopped/zero-balance behavior.
4. [RunPod current pricing page](https://www.runpod.io/pricing)
   - page displayed “Updated July 17, 2026” during this audit; GPU prices are dynamic and the console
     remains authoritative at deployment.

## AWS account, Free plan, India billing, and identity security

1. [Getting started with an AWS account](https://docs.aws.amazon.com/accounts/latest/reference/getting-started.html)
   - standalone signup fields, valid payment requirement, phone identity flow, support-plan choice,
     and activation time of minutes to as much as 24 hours.
2. [AWS Free Tier FAQs](https://aws.amazon.com/free/free-tier-faqs/)
   - current $100 signup/up-to-$100 additional credit model;
   - Free/Paid plan eligibility and valid payment requirement;
   - former/existing-customer ineligibility, non-transferable credits;
   - six-month/credit-exhaustion Free-plan end, 90-day retention after closure, paid upgrade needed to
     access/download, and 12-month credit expiry.
3. [Choosing an AWS Free Tier plan](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html)
   - official Free versus Paid comparison, selected-service limits, closure behavior, and upgrade
     effects.
4. [AWS Free Tier terms](https://aws.amazon.com/free/terms/)
   - binding eligibility, plan, credit, and expiration terms; these override a summary if changed.
5. [2025 announcement of the current Free Tier model](https://aws.amazon.com/about-aws/whats-new/2025/07/aws-free-tier-credits-month-free-plan/)
   - launch of up-to-$200 credits and six-month Free plan.
6. [AWS India billing setup](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/manage-account-payment-aispl.html)
   - AWS India seller-of-record/tax settings, PAN/GST, and invoices.
7. [Managing AWS India payments](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/edit-aispl-payment-method.html)
   - supported card networks, net banking, UPI/UPI AutoPay, INR 2 eligible-card verification/refund,
     and payment rules.
8. [Root-user MFA](https://docs.aws.amazon.com/IAM/latest/UserGuide/enable-mfa-for-root.html)
   - root MFA requirement and setup methods.
9. [IAM security best practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
   - protect root, use temporary credentials/federation, MFA, and least privilege.
10. [EC2 instance launch parameters](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-launch-parameters.html)
    - for accounts created on/after July 15, 2025, the current six-month/credit-based Free Tier
      includes `t3.small` among the listed instance types; usage still consumes credits.

## Amazon S3 creation, security, pricing, and integrity

1. [Creating a general-purpose S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)
   - immutable bucket name/region choices, Object Ownership/ACLs, Block Public Access, encryption,
     versioning, and creation workflow.
2. [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/)
   - current storage classes, storage/request/retrieval/transition/management pricing and examples.
3. [Official AWS Price List bulk API](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html)
   and the [S3 `us-east-2` current offer file](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-2/index.json)
   - machine-readable values rechecked on 2026-07-23: S3 Standard $0.023/GB-month first 50 TB,
     PUT/COPY/POST/LIST $0.005/1,000, GET/other $0.004/10,000, Standard-IA $0.0125/GB-month,
     Deep Archive transition $0.05/1,000.
4. [AWS global network/data-transfer FAQ](https://aws.amazon.com/about-aws/global-infrastructure/global-network/faqs/)
   - internet ingress free;
   - same-region direct S3↔EC2 transfer free;
   - 100 GB/month global data-transfer-out allowance;
   - migration-off-AWS credit request, preapproval, 60-day window, good-standing/deletion conditions.
5. [EC2 On-Demand data-transfer pricing](https://aws.amazon.com/ec2/pricing/on-demand/#Data_Transfer)
   - aggregated free outbound allowance and relevant transfer categories.
6. [AWS Data Transfer `us-east-2` current offer file](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSDataTransfer/current/us-east-2/index.json)
   - machine-readable $0.09/GB first 10 TB/month outbound after the global free tier, then lower tiers.
7. [Checking object integrity during upload/download](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html)
   - checksum validation and why checksum algorithms/ETags must be interpreted correctly.
8. [Checking object integrity at rest](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-at-rest.html)
   - stored checksum and S3 Batch checksum options/cost implications.
9. [S3 Inventory](https://docs.aws.amazon.com/AmazonS3/latest/userguide/configure-inventory.html)
   - scheduled destination-side object inventories and fields.
10. [Using S3 Versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html)
    - protection model and billing of versions.
11. [S3 storage classes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/storage-class-intro.html)
    - Standard-IA minimum size/duration/retrieval and Glacier durability/access rules.
12. [S3 Glacier storage classes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/glacier-storage-classes.html)
    - Deep Archive 180-day minimum, 9–48 hour retrieval modes, and 40 KB metadata overhead/object.
13. [Amazon S3 Glacier FAQ](https://aws.amazon.com/s3/faqs/)
    - Deep Archive advertised starting price of $0.00099/GB-month and retrieval/cost components.
14. [Working with archived objects](https://docs.aws.amazon.com/AmazonS3/latest/userguide/archived-objects.html)
    - temporary restored copies, restore windows, and Batch Operations.
15. [July 16, 2026 Standard-IA transition update](https://aws.amazon.com/about-aws/whats-new/2026/07/s3-removes-30-day-transitions-standard-ia-one-zone-ia/)
    - lifecycle may transition immediately into Standard-IA/One Zone-IA; storage-class economics still
      apply.
16. [AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html)
    - alerting/forecasting controls; alerts are not a storage hard cap.

## AWS transfer-service compatibility research

1. [DataSync transfers with other-cloud object storage](https://docs.aws.amazon.com/datasync/latest/userguide/creating-other-cloud-object-location.html)
   - endpoint/credentials, Enhanced versus Basic mode, tags, costs, and other-cloud considerations.
2. [Third-party cloud transfer considerations](https://docs.aws.amazon.com/datasync/latest/userguide/third-party-cloud-transfer-considerations.html)
   - validated locations, API compatibility testing, and object-tag behavior.
3. [DataSync S3 permissions/operations](https://docs.aws.amazon.com/datasync/latest/userguide/create-s3-location.html)
   - bucket-location, list/multipart, object/tag/version, put/delete permissions used by S3 workflows.
4. [AWS DataSync pricing](https://aws.amazon.com/datasync/pricing/)
   - transfer/task/request/compute pricing if a compatible workflow is used.

RunPod is not listed as validated, and its official compatibility table lacks bucket-location and
tagging operations used/considered by DataSync. That is why this package does not claim DataSync is a
supported one-click route.

## Rclone primary documentation

Rclone is not AWS or RunPod, but these are the transfer client’s own primary docs:

1. [Installation](https://rclone.org/install/)
2. [S3 backend](https://rclone.org/s3/)
   - AWS/Other provider configuration, endpoint/path style, EC2 IAM auth, hashes, multipart behavior,
     S3-to-S3 paths, and environment-config fields.
3. [Environment/config variables](https://rclone.org/docs/#environment-variables)
   - `RCLONE_CONFIG_<REMOTE>_<OPTION>` in-memory remote syntax.
4. [`rclone copy`](https://rclone.org/commands/rclone_copy/)
   - skips identical objects and does not delete destination extras.
5. [`rclone check`](https://rclone.org/commands/rclone_check/)
   - non-mutating size/hash comparison and `--download` byte-stream comparison.
6. [`rclone lsf`](https://rclone.org/commands/rclone_lsf/)
   - recursive machine-readable CSV path/size/hash inventories and unavailable-hash semantics.
7. [`rclone size`](https://rclone.org/commands/rclone_size/)
   - aggregate object count and logical bytes.

## Something-Something V2 publisher sources

1. [Qualcomm Something-Something V2 overview](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
   - 220,847 total videos, 168,913 train, 24,777 validation, 27,157 unlabeled test, 174 labels,
     12 fps, VP9 WebM distribution, 19.4 GB total download, and research-use availability.
2. [Qualcomm official download page](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads)
   - current instruction to download all files: the download-instruction document, both video ZIPs,
     and the labels package; each ZIP is unpacked separately before the documented archive extraction.
3. [Qualcomm research-use data license](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
   - the overview links the publisher's current Data License Agreement; use the agreement displayed
     to the authenticated account at download time.

The official download files require an interactive publisher workflow. This package intentionally
does not substitute a third-party mirror or claim stable signed download URLs/checksums. The operator
records the filenames, byte sizes, and SHA-256 values actually received in the rebuild manifest.

## EGO4D publisher sources

1. [EGO4D Start Here](https://ego4d-data.org/docs/start-here/)
   - license execution, typical approximately 48-hour approval, emailed AWS credentials, 14-day
     credential expiry, renewal expectation, and official CLI installation.
2. [EGO4D CLI tool](https://ego4d-data.org/docs/CLI/)
   - `pip install ego4d`, dataset/tier selection, `video_540ss`, video-UID filtering, master
     `ego4d.json`, per-dataset manifests, and destination layout.
3. [EGO4D videos](https://ego4d-data.org/docs/data/videos/)
   - canonical video/clip variants and the 540-short-side convention; canonical clips use constant
     30 fps. The HJEPA four-second/12-fps/256-short-side chunks are project-generated derivatives,
     not an EGO4D-published tier.
4. [EGO4D metadata](https://ego4d-data.org/docs/data/metadata/)
   - authoritative video metadata fields and release structure used when forming the deterministic
     project selection.
5. [Official EGO4D CLI repository](https://github.com/facebookresearch/Ego4d)
   - source implementation for the publisher CLI and manifest/download behavior.

The 210-hour selection, seed 42 split, four batches, 4-second H.264 chunks, 12 fps, 48 frames, CRF
27, shorter side 256, and no-audio contract come from this repository's primary EGO4D guide and
scripts—not from the EGO4D publisher. They are deliberately described as a project derivation.

## Hugging Face model and download sources

1. [Download files from the Hub](https://huggingface.co/docs/huggingface_hub/guides/download)
   - cache semantics, `snapshot_download`, and the requirement to use a full commit hash for an
     immutable revision.
2. [Hugging Face authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)
   - `hf auth login`, token storage, and authenticated downloads.
3. [Gated models](https://huggingface.co/docs/hub/models-gated)
   - access-request/approval behavior for gated repositories.
4. [V-JEPA2 model repository](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f)
   - exact project-pinned snapshot `b3c1679b7c34d3255ef3547f27c7b226aefab26f`.
5. [SigLIP2 model repository](https://huggingface.co/google/siglip2-base-patch16-256/tree/3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab)
   - exact project-pinned snapshot `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`.
6. [DINOv3 model repository](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/tree/5931719e67bbdb9737e363e781fb0c67687896bc)
   - exact project-pinned snapshot `5931719e67bbdb9737e363e781fb0c67687896bc` and gated-access
     status.

## Weights & Biases recovery semantics

1. [W&B Artifacts overview](https://docs.wandb.ai/models/artifacts)
   - a file is stored as an artifact only when code explicitly adds/logs it; metadata/metrics alone
     do not imply checkpoint-byte upload.
2. [Download and use artifacts](https://docs.wandb.ai/models/artifacts/download-and-use-an-artifact)
   - explicit artifact download workflow.
3. [Track external files](https://docs.wandb.ai/models/artifacts/track-external-files)
   - reference artifacts can store external-object metadata/references rather than the object bytes.
4. [`wandb sync`](https://docs.wandb.ai/models/ref/cli/wandb-sync)
   - uploads an existing local offline/incomplete run directory or `.wandb` file; it cannot recreate
     a local queue after its bytes are deleted.
5. [`wandb offline`](https://docs.wandb.ai/models/ref/cli/wandb-offline)
   - offline mode saves data locally until a later sync.
6. [`Run.save`](https://docs.wandb.ai/ref/python/experiments/run/#method-runsave)
   - files are uploaded only through an explicit path/glob save call and its policy.
7. [`wandb restore`](https://docs.wandb.ai/models/ref/cli/wandb-restore)
   - can reconstruct recorded config and, when available, Git diff/Docker state; it is not a general
     filesystem/checkpoint restore mechanism.

## Project-owned primary sources

These establish what the volume must contain and how readiness is tested:

- `AGENT_FILES/AGENTS.md`
- `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`
- `AGENT_FILES/SETUPS/SETUP.md`
- `AGENT_FILES/SETUPS/NEW_POD.md`
- `AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`
- `AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md`
- `AGENT_FILES/GUIDE_AUTONOMOUS_REMOTE_RUN.md`
- current `config.py`, `data.py`, `provenance.py`, `train.py`, `encoders.py`, `whiten_stats.py`,
  dataset scripts, probe scripts, tests, and all KANBAN experiment `GUIDE.md` files.

The cleaned result of the repository-wide `/workspace` reference audit is preserved in
[`02_CODE_REFERENCED_PATH_CATALOG.md`](02_CODE_REFERENCED_PATH_CATALOG.md).
