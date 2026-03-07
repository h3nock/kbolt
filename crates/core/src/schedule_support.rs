use kbolt_types::KboltError;

use crate::Result;

pub(crate) fn schedule_id_number(id: &str) -> Option<u32> {
    let number = id.strip_prefix('s')?.parse::<u32>().ok()?;
    if number == 0 {
        return None;
    }
    Some(number)
}

pub(crate) fn schedule_id_sort_key(id: &str) -> u32 {
    schedule_id_number(id).unwrap_or(u32::MAX)
}

pub(crate) fn parse_canonical_schedule_time(time: &str) -> Result<(u32, u32)> {
    let (hour, minute) = time
        .split_once(':')
        .ok_or_else(|| KboltError::InvalidInput(format!("invalid schedule time: {time}")))?;
    let hour = hour
        .parse::<u32>()
        .map_err(|_| KboltError::InvalidInput(format!("invalid schedule time: {time}")))?;
    let minute = minute
        .parse::<u32>()
        .map_err(|_| KboltError::InvalidInput(format!("invalid schedule time: {time}")))?;

    if hour > 23 || minute > 59 {
        return Err(KboltError::InvalidInput(format!("invalid schedule time: {time}")).into());
    }

    Ok((hour, minute))
}

#[cfg(test)]
mod tests {
    use super::{parse_canonical_schedule_time, schedule_id_number, schedule_id_sort_key};

    #[test]
    fn schedule_id_number_rejects_invalid_or_zero_ids() {
        assert_eq!(schedule_id_number("s1"), Some(1));
        assert_eq!(schedule_id_number("s10"), Some(10));
        assert_eq!(schedule_id_number("s0"), None);
        assert_eq!(schedule_id_number("schedule-1"), None);
    }

    #[test]
    fn schedule_id_sort_key_sends_invalid_ids_to_end() {
        assert!(schedule_id_sort_key("invalid") > schedule_id_sort_key("s9"));
    }

    #[test]
    fn parse_canonical_schedule_time_rejects_out_of_range_values() {
        let err = parse_canonical_schedule_time("25:99").expect_err("invalid time should fail");
        assert!(err.to_string().contains("invalid schedule time: 25:99"));
    }
}
