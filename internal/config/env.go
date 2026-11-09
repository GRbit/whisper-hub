package config

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

var durationType = reflect.TypeFor[time.Duration]()

// envField is one settable leaf of Config with its variable name.
type envField struct {
	name string
	v    reflect.Value
}

// envFields lists every leaf setting of c, named from the YAML path.
func envFields(c *Config) []envField {
	var out []envField
	walk(reflect.ValueOf(c).Elem(), envPrefix, &out)
	return out
}

func walk(v reflect.Value, prefix string, out *[]envField) {
	t := v.Type()
	for i := range t.NumField() {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		tag := f.Tag.Get("yaml")
		name, opts, _ := strings.Cut(tag, ",")
		if name == "-" {
			continue
		}
		fv := v.Field(i)
		if opts == "inline" {
			walk(fv, prefix, out)
			continue
		}
		if name == "" {
			name = strings.ToLower(f.Name)
		}
		full := prefix + strings.ToUpper(name)
		if f.Type.Kind() == reflect.Struct && f.Type != durationType {
			walk(fv, full+"_", out)
			continue
		}
		*out = append(*out, envField{name: full, v: fv})
	}
}

func applyEnv(c *Config, getenv func(string) (string, bool)) error {
	for _, f := range envFields(c) {
		val, ok := getenv(f.name)
		if !ok {
			continue
		}
		if err := setValue(f.v.Addr().Interface(), val); err != nil {
			return fmt.Errorf("config: %s: %w", f.name, err)
		}
	}
	return nil
}

// setValue parses s into the variable dst points to.
func setValue(dst any, s string) error {
	v := reflect.ValueOf(dst).Elem()
	s = strings.TrimSpace(s)
	if v.Type() == durationType {
		d, err := time.ParseDuration(s)
		if err != nil {
			return err
		}
		v.SetInt(int64(d))
		return nil
	}
	switch v.Kind() {
	case reflect.String:
		v.SetString(s)
	case reflect.Bool:
		b, err := strconv.ParseBool(s)
		if err != nil {
			return err
		}
		v.SetBool(b)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return err
		}
		v.SetInt(n)
	case reflect.Float32, reflect.Float64:
		n, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return err
		}
		v.SetFloat(n)
	case reflect.Slice:
		if v.Type().Elem().Kind() == reflect.String && !strings.HasPrefix(s, "[") {
			v.Set(reflect.ValueOf(splitList(s)).Convert(v.Type()))
			return nil
		}
		return yamlInto(v, s)
	case reflect.Map:
		if v.Type().Key().Kind() == reflect.String && v.Type().Elem().Kind() == reflect.String && !strings.HasPrefix(s, "{") {
			m := reflect.MakeMap(v.Type())
			for _, kv := range splitList(s) {
				k, val, ok := strings.Cut(kv, "=")
				if !ok {
					return fmt.Errorf("%q is not key=value", kv)
				}
				m.SetMapIndex(reflect.ValueOf(strings.TrimSpace(k)), reflect.ValueOf(strings.TrimSpace(val)))
			}
			v.Set(m)
			return nil
		}
		return yamlInto(v, s)
	default:
		return yamlInto(v, s)
	}
	return nil
}

func splitList(s string) []string {
	out := []string{}
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func yamlInto(v reflect.Value, s string) error {
	fresh := reflect.New(v.Type())
	dec := yaml.NewDecoder(strings.NewReader(s))
	dec.KnownFields(true)
	if err := dec.Decode(fresh.Interface()); err != nil {
		return err
	}
	v.Set(fresh.Elem())
	return nil
}
