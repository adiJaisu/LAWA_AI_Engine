const TARGET_TIME_ZONE = "America/Chicago";
const DATE_ONLY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

const targetOffsetFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: TARGET_TIME_ZONE,
  timeZoneName: "shortOffset",
  hour: "2-digit",
});

const dateTimeFormatter = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: true,
  timeZone: TARGET_TIME_ZONE,
});

const dateFormatter = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  timeZone: TARGET_TIME_ZONE,
});

const parseValidDate = (value?: string | number | Date | null) => {
  if (!value) return null;

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};

const getDateParts = (date: Date) => ({
  year: date.getFullYear(),
  month: String(date.getMonth() + 1).padStart(2, "0"),
  day: String(date.getDate()).padStart(2, "0"),
  hours: String(date.getHours()).padStart(2, "0"),
  minutes: String(date.getMinutes()).padStart(2, "0"),
  seconds: String(date.getSeconds()).padStart(2, "0"),
  milliseconds: date.getMilliseconds(),
});

const parseShortOffsetToMinutes = (offsetLabel: string) => {
  if (offsetLabel === "GMT" || offsetLabel === "UTC") {
    return 0;
  }

  const match = offsetLabel.match(/^GMT([+-])(\d{1,2})(?::?(\d{2}))?$/);
  if (!match) {
    return 0;
  }

  const [, sign, hours, minutes = "00"] = match;
  const totalMinutes = Number(hours) * 60 + Number(minutes);
  return sign === "+" ? totalMinutes : -totalMinutes;
};

const getTargetOffsetMinutes = (value: Date) => {
  const offsetLabel =
    targetOffsetFormatter.formatToParts(value).find((part) => part.type === "timeZoneName")
      ?.value ?? "GMT";

  return parseShortOffsetToMinutes(offsetLabel);
};

const formatOffset = (offsetMinutes: number) => {
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const absoluteMinutes = Math.abs(offsetMinutes);
  const hours = String(Math.floor(absoluteMinutes / 60)).padStart(2, "0");
  const minutes = String(absoluteMinutes % 60).padStart(2, "0");
  return `${sign}${hours}:${minutes}`;
};

export const toTargetTimeZoneInstant = (value?: string | number | Date | null) => {
  const sourceDate = parseValidDate(value);
  if (!sourceDate) return null;

  const { year, month, day, hours, minutes, seconds, milliseconds } = getDateParts(sourceDate);

  let utcMillis = Date.UTC(
    year,
    Number(month) - 1,
    Number(day),
    Number(hours),
    Number(minutes),
    Number(seconds),
    milliseconds
  );

  for (let i = 0; i < 2; i += 1) {
    const offsetMinutes = getTargetOffsetMinutes(new Date(utcMillis));
    utcMillis = Date.UTC(
      year,
      Number(month) - 1,
      Number(day),
      Number(hours),
      Number(minutes),
      Number(seconds),
      milliseconds
    )
      - offsetMinutes * 60 * 1000;
  }

  return new Date(utcMillis);
};

export const toTargetTimeZoneApiDateTime = (value?: string | number | Date | null) => {
  const sourceDate = parseValidDate(value);
  if (!sourceDate) return undefined;

  const targetInstant = toTargetTimeZoneInstant(sourceDate);
  if (!targetInstant) return undefined;

  const offsetMinutes = getTargetOffsetMinutes(targetInstant);
  const offset = formatOffset(offsetMinutes);
  const { year, month, day, hours, minutes, seconds } = getDateParts(sourceDate);

  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}${offset}`;
};

export const toApiDate = (value?: string | number | Date | null) => {
  const sourceDate = parseValidDate(value);
  if (!sourceDate) return undefined;

  const { year, month, day } = getDateParts(sourceDate);
  return `${year}-${month}-${day}`;
};

export const formatDateTime = (
  value?: string | number | Date | null
): string => {
  const date = parseValidDate(value);
  return date ? dateTimeFormatter.format(date) : "NA";
};

export const formatDate = (
  value?: string | number | Date | null
): string => {
  if (!value) return "NA";

  if (typeof value === "string" && DATE_ONLY_PATTERN.test(value.trim())) {
    const [year, month, day] = value.trim().split("-");
    return `${month}/${day}/${year}`;
  }

  const date = parseValidDate(value);
  return date ? dateFormatter.format(date) : "NA";
};

export const formatDuration = (seconds: number) => {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0s";
  }

  const totalSeconds = Math.floor(seconds);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;

  const parts: string[] = [];

  if (hours > 0) {
    parts.push(`${hours}h`);
  }

  if (minutes > 0 || hours > 0) {
    parts.push(`${minutes}m`);
  }

  parts.push(`${remainingSeconds}s`);

  return parts.join(" ");
};
