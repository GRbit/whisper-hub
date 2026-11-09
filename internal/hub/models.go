package hub

import (
	"context"
	"slices"

	"github.com/GRbit/whisper-hub/internal/core"
)

// modelView merges what the healthy workers serving one model id report.
// Capabilities are ANDed so a job is valid on whichever worker it lands.
type modelView struct {
	id        string
	engine    string
	languages []string
	caps      core.Capabilities
}

func (h *Hub) servedLocked(id string) bool {
	for _, w := range h.workers {
		if w.serves(id) {
			return true
		}
	}
	return false
}

func (h *Hub) viewLocked(id string) (modelView, bool) {
	v := modelView{id: id}
	found := false
	for _, w := range h.workers {
		if !w.serves(id) {
			continue
		}
		if !found {
			v.engine, v.languages, v.caps = w.info.Engine, w.info.Languages, w.info.Capabilities
			found = true
			continue
		}
		v.caps.WordTimestamps = v.caps.WordTimestamps && w.info.Capabilities.WordTimestamps
		v.caps.VADFilter = v.caps.VADFilter && w.info.Capabilities.VADFilter
		v.caps.Diarize = v.caps.Diarize && w.info.Capabilities.Diarize
		v.caps.Translate = v.caps.Translate && w.info.Capabilities.Translate
	}
	return v, found
}

// defaultModelLocked returns the configured default, else the default of the
// first healthy worker in config order, else "".
func (h *Hub) defaultModelLocked() string {
	if h.cfg.DefaultModel != "" {
		return h.cfg.DefaultModel
	}
	for _, w := range h.workers {
		if w.healthy && w.info != nil {
			if d := w.info.DefaultModel(); d != "" {
				return d
			}
		}
	}
	return ""
}

// resolveLocked maps a client model name to a served model id and checks
// access. The order is: empty -> default, served id, alias, a known but
// currently unserved id (unavailable, never a silent fallback), then the
// unknown-model policy.
func (h *Hub) resolveLocked(ctx context.Context, name string) (modelView, error) {
	def := h.defaultModelLocked()
	id := ""
	switch target, isAlias := h.cfg.Aliases[name]; {
	case name == "":
		id = def
	case h.servedLocked(name):
		id = name
	case isAlias:
		id = target
		if id == "" {
			id = def
		}
	case h.known[name] || name == h.cfg.DefaultModel:
		return modelView{}, core.Errorf(core.KindUnavailable, "model %q is not available right now", name)
	case h.cfg.UnknownModel == UnknownModelError:
		return modelView{}, core.Errorf(core.KindModelNotFound, "model %q not found", name)
	default:
		h.log.Debug("unknown model, using default", "requested", name, "default", def)
		id = def
	}
	if id == "" {
		return modelView{}, core.Errorf(core.KindUnavailable, "no engine worker available")
	}
	v, ok := h.viewLocked(id)
	if !ok {
		return modelView{}, core.Errorf(core.KindUnavailable, "model %q is not available right now", id)
	}
	if h.authz != nil && !h.authz.AllowModel(core.IdentityFrom(ctx), id, id == def) {
		shown := name
		if shown == "" {
			shown = id
		}
		return modelView{}, core.Errorf(core.KindForbidden, "model %q is not allowed for this user", shown)
	}
	return v, nil
}

// Models lists the served models the caller may use, default first, then
// by id.
func (h *Hub) Models(ctx context.Context) []core.Model {
	id := core.IdentityFrom(ctx)
	h.mu.Lock()
	defer h.mu.Unlock()
	def := h.defaultModelLocked()
	var ids []string
	for _, w := range h.workers {
		if !w.healthy || w.info == nil {
			continue
		}
		for _, m := range w.info.Models {
			if !slices.Contains(ids, m.ID) {
				ids = append(ids, m.ID)
			}
		}
	}
	slices.SortFunc(ids, func(a, b string) int {
		switch {
		case a == def:
			return -1
		case b == def:
			return 1
		case a < b:
			return -1
		case a > b:
			return 1
		}
		return 0
	})
	out := make([]core.Model, 0, len(ids))
	for _, m := range ids {
		if h.authz != nil && !h.authz.AllowModel(id, m, m == def) {
			continue
		}
		v, _ := h.viewLocked(m)
		var aliases []string
		for name, target := range h.cfg.Aliases {
			// A served id shadows an alias of the same name in resolveLocked.
			if (target == m || (target == "" && m == def)) && !slices.Contains(ids, name) {
				aliases = append(aliases, name)
			}
		}
		slices.Sort(aliases)
		out = append(out, core.Model{
			ID: m, Aliases: aliases, Engine: v.engine, Languages: slices.Clone(v.languages), Default: m == def,
			Capabilities: v.caps,
		})
	}
	return out
}
