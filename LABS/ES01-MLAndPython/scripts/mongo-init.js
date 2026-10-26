const appUser = process.env.MONGO_APP_USERNAME;
const appPassword = process.env.MONGO_APP_PASSWORD;
const databaseName = process.env.MONGO_INITDB_DATABASE;

if (!appUser || !appPassword || !databaseName) {
  throw new Error("MongoDB application user configuration is incomplete");
}

const applicationDatabase = db.getSiblingDB(databaseName);
if (!applicationDatabase.getUser(appUser)) {
  applicationDatabase.createUser({
    user: appUser,
    pwd: appPassword,
    roles: [{ role: "readWrite", db: databaseName }],
  });
}
