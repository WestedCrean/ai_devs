import pathlib
from enum import Enum
import pydantic
import polars as pl
from loguru import logger

from src.ai_devs_core import AIDevsClient, Config, get_config, JobClient, BatchJobConfig

TASK_NAME = "people"
DATA_SAVE_PATH = pathlib.Path("./data")


class Tag(Enum):
    IT = "IT"
    TRANSPORT = "transport"
    EDUKACJA = "edukacja"
    MEDYCYNA = "medycyna"
    PRACA_Z_LUDZMI = "praca z ludźmi"
    PRACA_Z_POJAZDAMI = "praca z pojazdami"
    PRACA_FIZYCZNA = "praca fizyczna"


class Classification(pydantic.BaseModel):
    classification: int
    tags: list[str]


def func_generating_dict(row):
    """Generate messages dict with id and job for Mistral batch API"""
    return [
        {
            "role": "system",
            "content": "You are a classification assistant. Analyze the job description and return ONLY a valid JSON response with the exact schema: {'classification': int, 'tags': list[str]}. Set classification to 1 if the person works in transportation, 0 if not, -1 if unsure. Use tags from this list only: ['IT', 'transport', 'edukacja', 'medycyna', 'praca z ludźmi', 'praca z pojazdami', 'praca fizyczna']. Never add explanations or text outside the JSON.",
        },
        {
            "role": "user",
            "content": f"Analyze this job description and return JSON only:\n\nID: {row['id']}\nJob: {row['job']}",
        },
    ]


def main():
    # Initialize configuration and clients
    config: Config = get_config()
    logger.info("Configuration loaded.")

    ai_devs_core = AIDevsClient(
        api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
    )
    job_client = JobClient(config)

    # Read and filter data
    logger.info("Reading and filtering data...")
    df = ai_devs_core.get_dataset(dataset=TASK_NAME, save_path=DATA_SAVE_PATH)
    logger.info(f"Initial data count: {len(df)}")
    logger.info(df)

    # Calculate age and filter
    df = df.select(
        pl.all(),
        pl.col("birthDate").str.to_datetime("%Y-%m-%d").dt.year().alias("born"),
        (
            pl.datetime(2024, 1, 1).dt.year()
            - pl.col("birthDate").str.to_datetime("%Y-%m-%d").dt.year()
        ).alias("age"),
    )

    df = df.with_row_index("id").filter(
        pl.col("gender") == "M",
        pl.col("age") >= 20,
        pl.col("age") <= 40,
        pl.col("birthPlace") == "Grudziądz",
    )
    logger.info(f"Filtered data count: {len(df)}")

    # Prepare data for batch processing
    df_with_messages = df.select(["id", "job"])
    logger.info(f"Prepared {len(df_with_messages)} records for batch processing")

    # Run batch job with enhanced configuration
    logger.info("Starting batch job...")
    result_df = job_client.batch_job(
        df=df_with_messages,
        schema=Classification,
        task=TASK_NAME,
        message_generator=func_generating_dict,
        config=BatchJobConfig(
            model="mistral-small-latest",
            poll_interval=5,
            timeout=120,
            max_workers=1,  # Set to >1 for parallel processing
            max_retries=5,
            chunk_size=1000,
            rate_limit=0,  # 0 = unlimited
        ),
    )

    # Log results
    logger.info("Batch job completed!")
    logger.info(f"Result DataFrame shape: {result_df.shape}")
    logger.info(f"Result columns: {result_df.columns}")
    logger.info("Sample results:")
    logger.info(
        result_df.select(["id", "job", "classification", "tags", "_success"]).limit(10)
    )

    # Summary statistics
    success_count = result_df.filter(pl.col("_success")).height
    logger.info(f"Successfully processed: {success_count}/{len(result_df)}")

    # Log metrics
    metrics = job_client.get_metrics()
    logger.info(f"Processing metrics: {metrics}")

    final_df = (
        df.join(result_df, on="id")
        .filter(pl.col("classification") == 1)
        .select(
            pl.col("name"),
            pl.col("surname"),
            pl.col("gender"),
            pl.col("born"),
            pl.col("birthPlace").alias("city"),
            pl.col("tags"),
        )
    )
    ai_devs_core.save_lesson_output(lesson_code="s01e01", df=final_df)

    res = ai_devs_core.verify(task=TASK_NAME, data=final_df.to_dicts())
    logger.info(f"Response from AI_devs API: {res}")
    return


if __name__ == "__main__":
    main()
