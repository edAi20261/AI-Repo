# Schema the teacher persona needs

**These are requests to the backend, not work this repo can do.**

Every `ai.*` table is declared in `backend/prisma/schema.prisma` and owned by the
backend's Prisma migrations. Raw SQL applied from here would be reverted by the
next `prisma migrate`, so each table is written as the Prisma model plus the SQL
it produces. `scripts/dev_ai_schema.sql` carries the same SQL for a local
database.

Until both tables exist, keep `persona.enabled: false`. With it off, a chat turn
that names a teacher is answered by the default tutor and says so in the log, and
the `/ai/personas` endpoints answer `503`.

---

## 1. `ai.teacher_personas`: one row per version

A teacher may build many versions and approve one. At most one is `active` per
teacher, which is what the partial unique index enforces: the service retires
the old version before it activates the new one, in one transaction.

`consented_at` is when the teacher agreed to have a persona built from their
material. The platform keeps the agreement itself; this is the timestamp that
ties it to the version it allowed.

```prisma
model TeacherPersona {
  id            String   @id @default(dbgenerated("gen_random_uuid()")) @map("persona_id") @db.Uuid
  teacherId     String   @map("teacher_id") @db.Uuid
  version       Int
  status        String   @default("draft") @db.VarChar(20) // draft | active | inactive
  teacherName   String   @map("teacher_name") @db.VarChar(200)
  card          Json
  exemplarCount Int      @default(0) @map("exemplar_count")
  sources       Json     @default("{}")
  consentedAt   DateTime @map("consented_at") @db.Timestamptz(6)
  createdAt     DateTime @default(now()) @map("created_at") @db.Timestamptz(6)
  updatedAt     DateTime @default(now()) @map("updated_at") @db.Timestamptz(6)
  activatedAt   DateTime? @map("activated_at") @db.Timestamptz(6)

  user      User                     @relation(fields: [teacherId], references: [id], onDelete: Cascade)
  exemplars TeacherPersonaExemplar[]

  @@unique([teacherId, version], map: "teacher_personas_version_key")
  @@map("teacher_personas")
  @@schema("ai")
}
```

Prisma cannot declare a partial index, so the one-active rule goes in the
migration by hand:

```sql
CREATE TABLE "ai"."teacher_personas" (
    "persona_id"     uuid         NOT NULL DEFAULT gen_random_uuid(),
    "teacher_id"     uuid         NOT NULL,
    "version"        integer      NOT NULL,
    "status"         varchar(20)  NOT NULL DEFAULT 'draft',
    "teacher_name"   varchar(200) NOT NULL,
    "card"           jsonb        NOT NULL,
    "exemplar_count" integer      NOT NULL DEFAULT 0,
    "sources"        jsonb        NOT NULL DEFAULT '{}',
    "consented_at"   timestamptz(6) NOT NULL,
    "created_at"     timestamptz(6) NOT NULL DEFAULT now(),
    "updated_at"     timestamptz(6) NOT NULL DEFAULT now(),
    "activated_at"   timestamptz(6),
    CONSTRAINT "teacher_personas_pkey" PRIMARY KEY ("persona_id"),
    CONSTRAINT "teacher_personas_status_check"
        CHECK ("status" IN ('draft', 'active', 'inactive'))
);
CREATE UNIQUE INDEX "teacher_personas_version_key"
    ON "ai"."teacher_personas" ("teacher_id", "version");
CREATE UNIQUE INDEX "teacher_personas_one_active"
    ON "ai"."teacher_personas" ("teacher_id") WHERE "status" = 'active';
ALTER TABLE "ai"."teacher_personas"
    ADD CONSTRAINT "teacher_personas_teacher_id_fkey" FOREIGN KEY ("teacher_id")
    REFERENCES "identity"."users"("id") ON DELETE CASCADE;
```

The foreign key points at `User` (`identity.users`), the same way `ai.user_memory` does.

## 2. `ai.teacher_persona_exemplars`: the teacher's own words

These are the passages kept as examples of how the teacher explains. The tutor
is shown the nearest ones on each turn. `embedding` is the passage's vector from
the same model that embeds the syllabus, so it is compared with a student's
question in the service. A teacher holds at most 40 of these
(`persona.max_exemplars`), so no vector index is needed.

Deleting a version deletes its examples (`ON DELETE CASCADE`), and deleting the
teacher deletes both.

```prisma
model TeacherPersonaExemplar {
  id        String  @id @default(dbgenerated("gen_random_uuid()")) @map("exemplar_id") @db.Uuid
  personaId String  @map("persona_id") @db.Uuid
  ordinal   Int
  text      String
  prompt    String  @default("")
  move      String  @default("") @db.VarChar(200)
  sourceId  String  @map("source_id") @db.VarChar(120)
  embedding Float[] @db.Real

  persona TeacherPersona @relation(fields: [personaId], references: [id], onDelete: Cascade)

  @@unique([personaId, ordinal], map: "teacher_persona_exemplars_ordinal_key")
  @@map("teacher_persona_exemplars")
  @@schema("ai")
}
```

```sql
CREATE TABLE "ai"."teacher_persona_exemplars" (
    "exemplar_id" uuid         NOT NULL DEFAULT gen_random_uuid(),
    "persona_id"  uuid         NOT NULL,
    "ordinal"     integer      NOT NULL,
    "text"        text         NOT NULL,
    "prompt"      text         NOT NULL DEFAULT '',
    "move"        varchar(200) NOT NULL DEFAULT '',
    "source_id"   varchar(120) NOT NULL,
    "embedding"   real[]       NOT NULL,
    CONSTRAINT "teacher_persona_exemplars_pkey" PRIMARY KEY ("exemplar_id")
);
CREATE UNIQUE INDEX "teacher_persona_exemplars_ordinal_key"
    ON "ai"."teacher_persona_exemplars" ("persona_id", "ordinal");
ALTER TABLE "ai"."teacher_persona_exemplars"
    ADD CONSTRAINT "teacher_persona_exemplars_persona_id_fkey" FOREIGN KEY ("persona_id")
    REFERENCES "ai"."teacher_personas"("persona_id") ON DELETE CASCADE;
```

## 3. The price row

`persona_build` joins the credit actions. It needs no schema change, because
`ai.credit_ledger.action` is a `varchar(50)`.
