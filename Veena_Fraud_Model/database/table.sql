-- CREATE TABLE claims (
--     id SERIAL PRIMARY KEY,
--     Month VARCHAR(20),
--     DayOfWeek VARCHAR(20),
--     Make VARCHAR(50),
--     AccidentArea VARCHAR(50),
--     Sex VARCHAR(10),
--     MaritalStatus VARCHAR(20),
--     Age INT,
--     PolicyType VARCHAR(50),
--     vehicle_category VARCHAR(50),
--     VehiclePrice VARCHAR(50),
--     NumberOfCars INT,
--     submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- ALTER TABLE claims RENAME COLUMN month TO "Month";
-- ALTER TABLE claims RENAME COLUMN day_of_week TO "DayOfWeek";
-- ALTER TABLE claims RENAME COLUMN make TO "Make";
-- ALTER TABLE claims RENAME COLUMN accident_area TO "AccidentArea";
-- ALTER TABLE claims RENAME COLUMN sex TO "Sex";
-- ALTER TABLE claims RENAME COLUMN marital_status TO "MaritalStatus";
-- ALTER TABLE claims RENAME COLUMN age TO "Age";
-- ALTER TABLE claims RENAME COLUMN policy_type TO "PolicyType";
-- ALTER TABLE claims RENAME COLUMN vehicle_category TO "VehicleCategory";
-- ALTER TABLE claims RENAME COLUMN vehicle_price TO "VehiclePrice";
-- ALTER TABLE claims RENAME COLUMN number_of_cars TO "NumberOfCars";
-- ALTER TABLE claims RENAME COLUMN submitted_at TO "submitted_at";
ALTER DATABASE fruad_apidb RENAME TO fraud_apidb;

