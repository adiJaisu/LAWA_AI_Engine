import csv
import os
from src.utils.DatabaseManager import DatabaseManager
from src.constant.constants import Constants
from src.utils.Logger import LoggingConfig

logger = LoggingConfig().setup_logging()

def migrate_csv_to_db(csv_path):
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}")
        return

    db_manager = DatabaseManager()
    
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                # Based on tailgate.py: employee_id, card_id, name, odc_no
                db_manager.add_employee(
                    emp_id=str(row['employee_id']),
                    card_id=str(row['card_id']),
                    name=str(row['name']),
                    odc_no=str(row['odc_no'])
                )
                count += 1
        
        logger.info(f"Database migration completed successfully. {count} records migrated.")
    except Exception as e:
        logger.error(f"Error during migration: {e}")

if __name__ == "__main__":
    csv_location = "/home/adi/Downloads/Tailgate(1)/Tailgate/data/record2.csv"
    migrate_csv_to_db(csv_location)
